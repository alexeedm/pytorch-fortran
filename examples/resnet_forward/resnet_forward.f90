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

program resnet_forward
    use torch_ftn
    use iso_fortran_env

    implicit none

    integer :: n
    type(torch_module) :: torch_mod
    type(torch_tensor_wrap) :: input_tensors
    type(torch_tensor) :: out_tensor

    real(real32) :: input(224, 224, 3, 10)
    real(real32), pointer :: output(:, :)

    character(:), allocatable :: filename
    integer :: arglen, stat

    if (command_argument_count() /= 1) then
        print *, "Need to pass a single argument: Pytorch model file name"
        stop
    end if

    call get_command_argument(number=1, length=arglen)
    allocate(character(arglen) :: filename)
    call get_command_argument(number=1, value=filename, status=stat)

    input = 1.0
    call input_tensors%create
    call input_tensors%add_array(input)
    call torch_mod%load(filename)
    call torch_mod%forward(input_tensors, out_tensor)
    call out_tensor%to_array(output)

    print *, output(1:5, 1)
end program
