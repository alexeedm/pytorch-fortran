! Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! MIT License
! 
! Permission is hereby granted, free of charge, to any person obtaining a
! copy of this software and associated documentation files (the "Software"),
! to deal in the Software without restriction, including without limitation
! the rights to use, copy, modify, merge, publish, distribute, sublicense,
! and/or sell copies of the Software, and to permit persons to whom the
! Software is furnished to do so, subject to the following conditions:
! 
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
! 
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
! DEALINGS IN THE SOFTWARE.

! Simple regression example similar to the C++ one:
! https://github.com/pytorch/examples/blob/master/cpp/regression/regression.cpp

module utils 

    use iso_fortran_env

    implicit none

    contains  

    subroutine eval_polynomial(coeffs, x, result)
        ! For convinience, we use 2D arrays for x and result
        ! just as they are used in the NN
        real(real32), intent(in)  :: coeffs(:), x(:,:)
        real(real32), intent(out) :: result(size(x,1),size(x,2))

        integer :: i, j
        
        !$acc data copyout(result)

        !$acc kernels
        result = 0.0
        !$acc end kernels

        ! Simple polynomial evaluation
        !$acc parallel loop gang vector
        do i=1,size(x,2)
            do j=1,size(coeffs)
                result(1,i) = result(1,i) + coeffs(j) * (x(1,i)**(j-1))
            end do
        end do

        !$acc end data
    end subroutine

end module

program polynomial
    use torch_ftn
    use utils

    implicit none

    integer, parameter :: batch_size = 1024
    integer, parameter :: poly_order = 4
    integer, parameter :: max_batch_id = 100000

#ifdef _OPENACC
    logical, parameter :: use_gpu = .true.
#else
    logical, parameter :: use_gpu = .false.
#endif
    
    type(torch_module)      :: torch_mod
    type(torch_tensor)      :: out_tensor, target_tensor
    type(torch_tensor_wrap) :: in_tensors

    real(real32) :: loss
    real(real32), dimension(1, batch_size) :: input, target
    real(real32), pointer :: output(:, :)
    real(real32) :: coeffs(poly_order)
    integer :: flag, batch_idx

    character(:), allocatable :: in_fname, out_fname
    integer :: arglen, stat

    if (command_argument_count() /= 2) then
        print *, "Need to pass a two argument: initial model file name, trained model file name"
        stop
    end if

    call get_command_argument(number=1, length=arglen)
    allocate(character(arglen) :: in_fname)
    call get_command_argument(number=1, value=in_fname, status=stat)

    call get_command_argument(number=2, length=arglen)
    allocate(character(arglen) :: out_fname)
    call get_command_argument(number=2, value=out_fname, status=stat)

    call random_number(coeffs)

    print *, "Importing initial Torch model"
    flag = 0
    if (use_gpu) then
        flag = module_use_device
    end if
    call torch_mod%load(in_fname, flag)
    call torch_mod%create_optimizer_sgd(0.1)
    call in_tensors%create

    !$acc data create (input, target) copyin(coeffs)

    !$acc host_data use_device(input)
    call in_tensors%add_array(input)
    !$acc end host_data

    !$acc host_data use_device(target)
    call target_tensor%from_array(target)
    !$acc end host_data

    print *, "Starting to train the model..."
    do batch_idx = 1, max_batch_id
        ! We also can generate the numbers directly on the device and avoid copies
        call random_number(input)
        !$acc update device(input)

        call eval_polynomial(coeffs, input, target)
        call torch_mod%train(in_tensors, target_tensor, loss)

        if (mod(batch_idx, 100) == 0) then
            print "(A,I6,A,F9.6)", "Batch ",batch_idx," loss is ",loss
        end if
        if (loss < 1e-4) exit
    end do

    if (batch_idx < max_batch_id) then
        print *, "Hit target accuracy"
    else
        print *, "Couldn't hit target accuracy, exiting"
        stop
    end if

    ! Evaluation
    call random_number(input)
    !$acc update device(input)
    call eval_polynomial(coeffs, input, target)

    call torch_mod%forward(in_tensors, out_tensor)
    call out_tensor%to_array(output)

    !$acc update host(target, output)
    loss = sum( (target-output)**2 ) / batch_size
    
    print *, target(1,1:4), output(1,1:4)
    print "(A,F9.6)", "Mean squared error of the trained model is ", loss

    !$acc end data

    ! Save trained model
    print "(A,A)", "Saving trained model to ", out_fname
    call torch_mod%save(out_fname)
end program
