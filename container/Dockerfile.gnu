FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gdb \
        less \
        vim && \
    rm -rf /var/lib/apt/lists/*

# CMake version 3.25.1
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        make \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.sh && \
    mkdir -p /usr/local && \
    /bin/sh /var/tmp/cmake-3.25.1-linux-x86_64.sh --prefix=/usr/local --skip-license && \
    rm -rf /var/tmp/cmake-3.25.1-linux-x86_64.sh
ENV PATH=/usr/local/bin:$PATH

# GNU compiler
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common && \
    apt-add-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++-9 \
        gcc-9 \
        gfortran-9 && \
    rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/g++ g++ $(which g++-9) 30 && \
    update-alternatives --install /usr/bin/gcc gcc $(which gcc-9) 30 && \
    update-alternatives --install /usr/bin/gcov gcov $(which gcov-9) 30 && \
    update-alternatives --install /usr/bin/gfortran gfortran $(which gfortran-9) 30

RUN conda install -c conda-forge -y pybind11
