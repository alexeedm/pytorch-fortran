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

FROM nvcr.io/nvidia/pytorch:21.09-py3

RUN conda install -c anaconda netcdf4

# NVIDIA HPC SDK version 21.9
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bc \
        debianutils \
        g++ \
        gcc \
        gfortran \
        libatomic1 \
        libnuma1 \
        openssh-client \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://developer.download.nvidia.com/hpc-sdk/21.9/nvhpc_2021_219_Linux_x86_64_cuda_11.4.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/nvhpc_2021_219_Linux_x86_64_cuda_11.4.tar.gz -C /var/tmp -z && \
    cd /var/tmp/nvhpc_2021_219_Linux_x86_64_cuda_11.4 && NVHPC_ACCEPT_EULA=accept NVHPC_DEFAULT_CUDA=11.4 NVHPC_INSTALL_DIR=/opt/nvidia/hpc_sdk NVHPC_SILENT=true ./install && \
    rm -rf /var/tmp/nvhpc_2021_219_Linux_x86_64_cuda_11.4 /var/tmp/nvhpc_2021_219_Linux_x86_64_cuda_11.4.tar.gz

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        vim less gdb cmake && \
    rm -rf /var/lib/apt/lists/*
