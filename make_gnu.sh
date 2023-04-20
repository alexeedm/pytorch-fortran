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
set -e

# This wrapper uses the GNU compilers to build 
# training and inference examples

# Default Pytorch containers seem to have Pytorch installed through conda
PY_SITE_PATH=$(find /opt/conda/lib/ -maxdepth 1 -name 'python?.*' -type d)/site-packages
CMAKE_PREFIX_PATH="${PY_SITE_PATH}/torch/share/cmake;${PY_SITE_PATH}/pybind11/share/cmake"
echo "Trying to use Python libraries from $PY_SITE_PATH"

CONFIG=Release
OPENACC=0

BUILD_PATH=$(pwd -P)/gnu/
INSTALL_PATH=${1:-$BUILD_PATH/install/}
mkdir -p $BUILD_PATH/build_proxy $BUILD_PATH/build_fortproxy $BUILD_PATH/build_example
# c++ wrappers 
(
    cd $BUILD_PATH/build_proxy 
    cmake -DOPENACC=$OPENACC -DCMAKE_BUILD_TYPE=$CONFIG -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST ../../src/proxy_lib
    cmake --build . --parallel
    make install
)

# fortran bindings
(
    export PATH=$NVPATH:$PATH 
    cd $BUILD_PATH/build_fortproxy
    cmake -DOPENACC=$OPENACC -DCMAKE_BUILD_TYPE=$CONFIG -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_PREFIX_PATH=$INSTALL_PATH/lib ../../src/f90_bindings/
    cmake --build . --parallel
    make install
)

# fortran examples
(
    export PATH=$NVPATH:$PATH 
    cd $BUILD_PATH/build_example
    cmake -DOPENACC=$OPENACC -DCMAKE_BUILD_TYPE=$CONFIG -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_Fortran_COMPILER=gfortran ../../examples/
    cmake --build . --parallel
    make install
)
