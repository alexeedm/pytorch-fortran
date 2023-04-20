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

program python_training
    use torch_ftn
    use iso_fortran_env

    implicit none

    integer :: n
    type(torch_pymodule) :: torch_pymod
    type(torch_tensor) :: t_out, t_target
    type(torch_tensor_wrap) :: tw_in

    real(real32) :: input(2, 3), target(2), factor
    real(real32), pointer :: output(:,:)
    real(real32) :: loss
    logical :: is_completed

    character(:), allocatable :: filename
    integer :: arglen, stat
    
    nullify(output)
    
    if (command_argument_count() /= 1) then
        print *, "Need to pass a single argument: Pytorch model python script name"
        stop
    end if

    call get_command_argument(number=1, length=arglen)
    allocate(character(arglen) :: filename)
    call get_command_argument(number=1, value=filename, status=stat)

    input(1,:) = 1.0
    input(2,:) = 2.0
    factor = 3.0

    call tw_in%create
    call tw_in%add_array(input)
    call tw_in%add_scalar(factor)
    call t_target%from_array(target)

    call torch_pymod%load(filename)
    ! will call Python function ftn_pytorch_forward(input) -> output
    call torch_pymod%forward(tw_in, t_out)
    call t_out%to_array(output)
    print *, output

    ! will call Python function ftn_pytorch_train(input, target) -> (is_completed, loss)
    is_completed = torch_pymod%train(tw_in, t_target, loss)
    print *, is_completed, loss

end program
