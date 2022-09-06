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

# The Fortran wrapper libs need to be built with the same compiler as the code using them
project(PyTorch-FortranWrapper LANGUAGES Fortran)
find_package(pytorch_proxy REQUIRED)

# Fortran settings
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod_files)
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(dialect "-ffree-form -std=f2008 -fimplicit-none -g -cpp")
    set(bounds "-fbounds-check")
endif()
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(dialect "-stand f08 -free -implicitnone -g -cpp")
    set(bounds "-check bounds")
endif()
if(CMAKE_Fortran_COMPILER_ID MATCHES "PGI|NVHPC")
    set(dialect "-Mfreeform -Mdclchk -Mstandard -Mallocatable=03 -g -Mpreprocess")
    set(bounds "-C")
endif()
if (OPENACC)
    find_package(OpenACC REQUIRED)
    set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} ${OpenACC_Fortran_FLAGS}")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenACC_Fortran_FLAGS}")
endif()

set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} ${dialect} ${bounds}")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${dialect}")

# Generate f90 binding for all different ranks from template
add_custom_command(
    OUTPUT torch_ftn.f90
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen.py 
                    ${CMAKE_CURRENT_SOURCE_DIR}/torch_ftn.f90.templ torch_ftn.f90 
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/torch_ftn.f90.templ ${CMAKE_CURRENT_SOURCE_DIR}/gen.py
)

add_library(pytorch_fort_proxy SHARED torch_ftn.f90)
target_link_libraries(pytorch_fort_proxy PUBLIC pytorch_proxy)
target_include_directories(pytorch_fort_proxy PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")
target_include_directories(pytorch_fort_proxy INTERFACE $<INSTALL_INTERFACE:include/mod_files>)
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    target_link_options(pytorch_fort_proxy INTERFACE "-Wl,--disable-new-dtags")
endif()

set(libdest "lib/pytorch_fort_proxy")
install(TARGETS pytorch_fort_proxy EXPORT pytorch_fort_proxy DESTINATION lib)
install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY} DESTINATION include)
install(FILES pytorch_fort_proxy-config.cmake DESTINATION ${libdest})
install(EXPORT pytorch_fort_proxy DESTINATION ${libdest})
