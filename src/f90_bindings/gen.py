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
 
import sys

if len(sys.argv) < 2:
    print(f'ERROR: Usage is {sys.argv[0]} infile.f90.templ outfile.f90')
    sys.exit(1)
template_file = sys.argv[1]
out_file = sys.argv[2]
if not template_file.endswith('f90.templ'):
    print(f'ERROR: wrong input file, got {template_file}.')
    sys.exit(1)
if not out_file.endswith('.f90'):
    print(f'ERROR: wrong output file, got {out_file}.')
    sys.exit(1)
print(f'INFO: Generating {out_file} from {template_file}.')

dtypes = dict(fp32=dict(fortran_type='real32', c_type='TORCH_FTN_TYPE_FP'))
dims = range(1,7)
all_dims_dtypes = [(dim, dt[0], dt[1]) for dim in dims for dt in dtypes.items()]


def make_dim_dt_lines(file, line):
    for i, (d, dt, _) in enumerate(all_dims_dtypes):
        file.write(line.rstrip().format(dim=d, dt=dt))
        file.write(', &\n' if i < len(all_dims_dtypes) -1 else '\n')

def make_block(file, block_str):
    for d, dt, dt_info in all_dims_dtypes:
        dims_shape = ','.join([':']*d)
        file.write(block_str.format(dim=d, dt=dt, 
            dt_fort=dt_info['fortran_type'], dims_shape=dims_shape, c_type=dt_info['c_type']))

with open(template_file) as inf:
    template_lines = inf.readlines()

iprint = lambda x, y: print(f'{x}: {y}')
with open(out_file, 'w') as outf:
    i = 0
    while i < len(template_lines):
        line = template_lines[i]
        if line.startswith('<%'):
            iprint(i, 'Replace line')
            make_dim_dt_lines(outf, line[2:])
            i += 1
        elif line.startswith('<<%'):
            iprint(i, 'Start replace block')
            i += 1
            block_str = ''
            line = template_lines[i]
            while not line.startswith('%>>'):
                if (i >= len(template_lines)):
                    print(f'ERROR: replace block not closed')
                    sys.exit(1)
                block_str += line
                i += 1
                line = template_lines[i]
            i += 1
            iprint(i, 'End replace block')
            make_block(outf, block_str)
        else:
            outf.write(line)
            i += 1
    iprint(i, f'INFO: Wrote {i} lines.')