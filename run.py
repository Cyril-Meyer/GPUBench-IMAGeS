import argparse
import index

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # avoid tensorflow information messages

parser = argparse.ArgumentParser()
parser.add_argument('--vram', type=float, default=None,
                    help='only run benchmarks with vram usage less or equal to this value')
parser.add_argument('--tensorflow', action='store_true',
                    help='run all TensorFlow flagged benchmarks')
parser.add_argument('--pytorch', action='store_true',
                    help='run all PyTorch flagged benchmarks')
parser.add_argument('--id', nargs='*', type=str,
                    help='run all specified benchmarks')
args = parser.parse_args()

print('GPUBench-IMAGeS')

if args.vram is None:
    max_vram = float('inf')
else:
    max_vram = args.vram

if args.tensorflow:
    for id in index.tensorflow:
        bench = index.benchmarks[id]()
        result = bench.mark()
        print(f'{id} : {result}')

if args.pytorch:
    raise NotImplementedError

for id in args.id:
    if id not in index.benchmarks.keys():
        print(f'ERROR: wrong benchmark id : {id}')
    else:
        bench = index.benchmarks[id]()
        result = bench.mark()
        print(f'{id} : {result}')
