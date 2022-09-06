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

cmake_minimum_required(VERSION 3.13)

# C++ library should be built with the same compiler as Pytorch
if (OPENACC)
    project(PyTorch-ProxyLib LANGUAGES CXX CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__TORCH_FTN_USE_CUDA")
else()
    project(PyTorch-ProxyLib LANGUAGES CXX)
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# CUDA architecture
if(NOT DEFINED CUDA_ARCH_LIST)
    set(CUDA_ARCH_LIST 7.0)
endif()
set(ENV{TORCH_CUDA_ARCH_LIST} ${CUDA_ARCH_LIST})
string(REPLACE "." "" CMAKE_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

add_library(pytorch_proxy SHARED torch_proxy.cpp)

# General settings
set_target_properties(pytorch_proxy PROPERTIES
                                  CXX_STANDARD  17
                                  CUDA_STANDARD 14
                                  POSITION_INDEPENDENT_CODE ON
                                  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCH_LIST})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

# We need libtorch_python library for integration with pybind11
# This library is not included in TORCH_LIBRARIES, so we need to manually include it
set(filtered_lib ${TORCH_LIBRARIES})
list(FILTER filtered_lib INCLUDE REGEX ".*libc10.so$")
string(REPLACE "libc10.so" "" torch_lib_dir ${filtered_lib})

# Includes
target_link_libraries(pytorch_proxy PRIVATE Python3::Python)
target_link_libraries(pytorch_proxy PRIVATE "${torch_lib_dir}/libtorch_python.so")
target_link_libraries(pytorch_proxy PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(pytorch_proxy PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")

# Installation settings
set(libdest "lib/pytorch_proxy")
install(TARGETS pytorch_proxy EXPORT pytorch_proxy DESTINATION lib)
install(FILES pytorch_proxy-config.cmake DESTINATION ${libdest})
install(EXPORT pytorch_proxy DESTINATION ${libdest})