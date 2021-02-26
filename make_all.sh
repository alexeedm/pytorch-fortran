# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

#!/bin/bash
set -ex

# This wrapper uses the NVHPC compiler to build 
# training and inference examples
#
# Need to kick off 2 separate cmake builds with different compilers:
# 1. with GCC, build pytorch C++ wrapper lib that exposes things to Fortran
# 2. with NVHPC, build fortran bindings that just bind(c) to built lib from (1)
# TODO: set path for nvhpc compilers (subshell?)

NVPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/compilers/bin/:$PATH
CMAKE_PREFIX_PATH=/opt/conda/lib/python3.8/site-packages/torch/share/cmake

CONFIG=Debug
OPENACC=

# List CUDA compute capabilities
TORCH_CUDA_ARCH_LIST=7.0

INST=${1:-$(pwd -P)/install}
mkdir -p build_proxy build_fortproxy build_example
# c++ wrappers 
(
    cd build_proxy 
    cmake -DCMAKE_INSTALL_PREFIX=$INST -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST ../src/proxy_lib
    cmake --build   . --config $CONFIG --parallel
    make install
)

# fortran bindings
(
    export PATH=$NVPATH:$PATH 
    cd build_fortproxy
    cmake -DOPENACC=$OPENACC -DCMAKE_Fortran_COMPILER=nvfortran -DCMAKE_INSTALL_PREFIX=$INST -DCMAKE_PREFIX_PATH=$INST/lib ../src/f90_bindings/
    cmake --build . --config $CONFIG --parallel
    make install
)

# fortran examples
(
    export PATH=$NVPATH:$PATH 
    cd build_example
    cmake -DOPENACC=$OPENACC -DCMAKE_Fortran_COMPILER=nvfortran -DCMAKE_INSTALL_PREFIX=$INST ../examples/
    cmake --build . --config $CONFIG --parallel
    make install
)
