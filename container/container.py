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

#!/usr/bin/env python

import argparse
import hpccm
import hpccm.building_blocks as hbb
import hpccm.primitives as hp
import logging

def add_common_packages(stage):
    stage += hbb.apt_get(ospackages=['vim', 'less', 'gdb'])
    stage += hbb.cmake(eula=True)

def build_nvhpc(stage, args):
    import requests
    import re

    # Trying to identify the latest Pytorch tag
    pytorch_tag = args.pytorch_tag
    if (not pytorch_tag):
        page = requests.get(r'https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch').text
        m = re.search('>\s*(\S+)\s*\(Latest\) Security Scan Results', page, flags=re.MULTILINE)
        if (m):
            pytorch_tag = m[1]
    
    if (not pytorch_tag):
        logging.critical('Could not determine the latest PyTorch container tag, please provide it via --pytorch-tag argument')

    logging.info(f'Using PyTorch tag: {pytorch_tag}')
    stage += hp.baseimage(image = f'nvcr.io/nvidia/pytorch:{pytorch_tag}', _distro='ubuntu20')
    
    nvhpc = hbb.nvhpc(eula=True, cuda_multi=False, environment=False)
    try:
        logging.info(f'Using Nvidia HPC SDK version: {nvhpc._nvhpc__version}')
    except:
        pass

    stage += nvhpc

    # 23.1 pulls 23.3 for some reason, we'll remove it
    if (nvhpc._nvhpc__version == '23.1'):
        stage += hp.shell(commands = [
            r'''sed -ie 's/, nvhpc-2023 (>= 23.1)//g' /var/lib/dpkg/status''',
            r'apt autoremove -y'
        ])

    # Remove NV HPC bundled CUDA and point the compilers to the Torch CUDA
    stage += hp.shell(commands = [
        r'''cuda_path=$(find /opt/nvidia/hpc_sdk/Linux_x86_64/*.*/cuda -maxdepth 1 -name '??.?' -type d)''',
        r'rm -r $cuda_path',
        r'ln -s /usr/local/cuda $cuda_path'
    ])
    logging.info(f'Cleaning up HPC SDK package to reduce the image size, '+
                 f'recommended to build with docker-squash (https://github.com/goldmann/docker-squash)')

    # Workaround for the libgomp OpenACC issue
    stage += hp.environment(variables = {
        'LD_PRELOAD' : f'$LD_PRELOAD:/opt/nvidia/hpc_sdk/Linux_x86_64/{nvhpc._nvhpc__version}/compilers/lib/libaccnotify.so'
    })

    add_common_packages(stage)

def build_gnu(stage, args):
    import requests
    import re
    import json

    # Trying to identify the latest Pytorch tag
    pytorch_tag = args.pytorch_tag
    if (not pytorch_tag):
        from natsort import natsorted
        try:
            response = json.loads(requests.get(r'https://registry.hub.docker.com/v2/repositories/pytorch/pytorch/tags').text)
            versions = [cont['name'] for cont in response['results'] if ('devel' in cont['name'] and 'nightly' not in cont['name'])]
            pytorch_tag = natsorted(versions)[-1]
        except:
            pass
    if (not pytorch_tag):
        logging.critical('Could not determine the latest PyTorch container tag, please provide it via --pytorch-tag argument')

    logging.info(f'Using PyTorch tag: {pytorch_tag}')

    stage += hp.baseimage(image = f'pytorch/pytorch:{pytorch_tag}', _distro='ubuntu20')
    add_common_packages(stage)
    stage += hbb.gnu(extra_repository=True, version='9')
    stage += hp.shell(commands = [
        'conda install -c conda-forge -y pybind11'
    ])

def build_intel(stage, args):
    import requests
    import re

    if (args.format == 'singularity'):
        logging.critical(f'Intel container needs Docker format. You may convert the Dockerfile to Singularity recipe later')

    if (not args.pytorch_tag):
        args.pytorch_tag = 'latest'

    stage += hp.baseimage(image = f'intel/intel-optimized-pytorch:{args.pytorch_tag}', _distro='ubuntu20', _as='build')
    add_common_packages(stage)
    # Intel pytorch container don't seem to use conda anymore
    stage += hbb.apt_get(ospackages=['python3-dev'])
    stage += hp.shell(commands = [
        'python -m pip install pybind11'
    ])

    oneapi_docker = requests.get(r'https://raw.githubusercontent.com/intel/oneapi-containers/master/images/docker/hpckit/Dockerfile.ubuntu-20.04').text
    oneapi_patched = re.sub('FROM ubuntu.*as build', '', oneapi_docker)
    oneapi_patched = re.sub('COPY third-party-programs\.txt.*', '', oneapi_patched)
    oneapi_patched = re.sub('intel-oneapi-python', '', oneapi_patched)
    oneapi_patched = re.sub('ENV CONDA_.*', '', oneapi_patched)
    oneapi_patched = re.sub('FROM ubuntu:.*', 'FROM build', oneapi_patched)
    oneapi_patched = re.sub('COPY --from=build.*', '', oneapi_patched)
    oneapi_patched = re.sub('ENV PYTHONPATH=.*', '', oneapi_patched)
    oneapi_patched = re.sub('/opt/intel/oneapi/intelpython/latest/bin:', '', oneapi_patched)
    oneapi_patched = re.sub('/opt/intel/oneapi/intelpython/latest/condabin:', '', oneapi_patched)
    oneapi_patched = re.sub(r'ENV PATH=(.*)', r'ENV PATH=$PATH:\1', oneapi_patched)
    oneapi_patched = re.sub(r'ENV LD_LIBRARY_PATH=(.*)', r'ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\1', oneapi_patched)

    stage += hp.raw(docker = oneapi_patched)

################################################

parser = argparse.ArgumentParser(description='Genererate recipe for the bindings build environment',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--format', type=str, default='docker',
                    choices=['docker', 'singularity'],
                    help='Container specification format')
parser.add_argument('--recipe-file', type=argparse.FileType('w'), default='./Dockerfile',
                    help='Output recipe file')
parser.add_argument('--build-bindings', action='store_true',
                    help='Build the binding in the container, by default only the development environment will be built')

subparsers = parser.add_subparsers(title='Compilers', dest='compiler')
subparsers.required = True

p_intel = subparsers.add_parser('intel', help='Intel compilers: icc and ifort')
p_intel.set_defaults(function=build_intel)
p_intel.add_argument('--pytorch-tag', type=str, default=None, \
                    help='Pytorch container tag, see https://hub.docker.com/r/intel/intel-optimized-pytorch, default is the latest available')

p_gnu = subparsers.add_parser('gnu', help='GNU compilers: gcc and gfortran')
p_gnu.add_argument('--pytorch-tag', type=str, default=None, \
                    help='Pytorch container tag, see https://hub.docker.com/r/pytorch/pytorch/, default is the latest available')
p_gnu.set_defaults(function=build_gnu)

p_nvhpc = subparsers.add_parser('nvhpc', help='NVHPC compilers: nvcc and nvfortran')
p_nvhpc.add_argument('--pytorch-tag', type=str, default=None, \
                    help='Pytorch container tag, see https://ngc.nvidia.com/catalog/containers, default is the latest available')
p_nvhpc.set_defaults(function=build_nvhpc)

################################################

class ExitOnExceptionHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        if record.levelno in (logging.ERROR, logging.CRITICAL):
            raise SystemExit(-1)

logging.basicConfig(handlers=[ExitOnExceptionHandler()], level=logging.INFO)

args = parser.parse_args()
hpccm.config.set_container_format(args.format)

Stage0 = hpccm.Stage()
args.function(Stage0, args)
print(Stage0, file=args.recipe_file)
