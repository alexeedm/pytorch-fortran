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

parser = argparse.ArgumentParser(description='Generate build environment for ICON')
parser.add_argument('--format', type=str, default='docker',
                    choices=['docker', 'singularity'],
                    help='Container specification format')
parser.add_argument('--pytorch-version', type=str, default='21.09', \
                    help='Pytorch container version, see https://ngc.nvidia.com/catalog/containers')
parser.add_argument('--cuda-version', default='11.4', help='CUDA version of the pytorch container')
parser.add_argument('--recipe-file', type=argparse.FileType('w'), default='-')

args = parser.parse_args()

hpccm.config.set_container_format(args.format)

Stage0 = hpccm.Stage()
Stage0 += hp.baseimage(image = f'nvcr.io/nvidia/pytorch:{args.pytorch_version}-py3', _distro='ubuntu20')
Stage0 += hp.shell(commands = ['conda install -c anaconda netcdf4'])
Stage0 += hbb.nvhpc(eula=True, cuda=args.cuda_version, cuda_multi=False, environment=False)
Stage0 += hbb.apt_get(ospackages=['vim less gdb cmake'])

print(Stage0, file=args.recipe_file)
