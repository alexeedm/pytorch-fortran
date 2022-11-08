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
import argparse
from collections import namedtuple
import re
import logging
import itertools

##############################################################################

class ExitOnExceptionHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        if record.levelno in (logging.ERROR, logging.CRITICAL):
            raise SystemExit(-1)

logging.basicConfig(handlers=[ExitOnExceptionHandler()], level=logging.INFO)

##############################################################################

parser = argparse.ArgumentParser(description='Preprocess Fortran files',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('in_file',  type=argparse.FileType('r'), help='Input template filename')
parser.add_argument('out_file', type=argparse.FileType('w'), help='Output code filename',  nargs='?', default='-')

args = parser.parse_args()
print(f'INFO: Generating {args.out_file.name} from {args.in_file.name}')

##############################################################################

Dtype = namedtuple('Dtype', ['name', 'fortran_id', 'fortran_prec', 'c_id', 'size'])
Dimensions = namedtuple('Dimensions', ['rank', 'shape'])

dtypes = [
    Dtype('fp32 ', 'real',    'real32', 'TORCH_FTN_TYPE_FP',  4),
    Dtype('fp64 ', 'real',    'real64', 'TORCH_FTN_TYPE_FP',  8),
    Dtype('int32', 'integer', 'int32',  'TORCH_FTN_TYPE_INT', 4),
    Dtype('int64', 'integer', 'int64',  'TORCH_FTN_TYPE_INT', 8)
]
dimensions = [ Dimensions(d, ','.join([':']*d)) for d in range(1,7) ]

replacement_map_all = { 'dtype' : dtypes, 'dims' : dimensions }

##############################################################################

line_trigger=r'<%'
block_trigger_on =f'<{line_trigger}'
block_trigger_off=r'%>>'

def make_replacement_map(names):
    if (names):
        my_map = {}
        for name in names.split(','):
            if name not in replacement_map_all:
                logging.fatal(f'Variable {name} has no replacement rules')
            my_map[name] = replacement_map_all[name]
        return my_map
    else:
        return replacement_map_all
    

def replace_line(replacement_map, line):
    names = replacement_map.keys()
    lists = list(replacement_map.values())

    new_lines = []
    for el in itertools.product(*lists):
        new_lines.append( line.format( **dict(zip(names, el)) ) )

    return ', &\n'.join(new_lines)+'\n'

def replace_block(replacement_map, lines):
    names = replacement_map.keys()
    lists = list(replacement_map.values())

    new_lines = []
    for el in itertools.product(*lists):
        for line in lines:
            new_lines.append( line.format( **dict(zip(names, el)) ) )
        new_lines.append('\n')

    return ''.join(new_lines)+'\n'

##############################################################################

template_lines = args.in_file.readlines()

line_regex        = re.compile(f'^\s*{line_trigger}      (?: \s*\( (.*) \) | )  (.*)$', re.X)
block_begin_regex = re.compile(f'^\s*{block_trigger_on}  (?: \s*\( (.*) \) | )  .*$',   re.X)
block_end_regex   = re.compile(f'^\s*{block_trigger_off}', re.X)


memorize_line = False
block = []
for line_num, line in zip(range(len(template_lines)), template_lines):
    
    m = re.search(line_regex, line)
    if (m):
        logging.info(f'Replacing line {line_num+1}')
        res = replace_line(make_replacement_map(m.group(1)), m.group(2))
        args.out_file.write(res)
        continue

    m = re.search(block_begin_regex, line)
    if (m):
        logging.info(f'Replacing block from line {line_num+1}')
        my_map = make_replacement_map(m.group(1))
        memorize_line = True
        continue

    m = re.search(block_end_regex, line)
    if (m):
        logging.info(f'                till line {line_num+1}')
        res = replace_block(my_map, block)
        args.out_file.write(res)
        block.clear()
        memorize_line = False
        continue

    if (memorize_line):
        block.append(line)
        continue

    args.out_file.write(line)

