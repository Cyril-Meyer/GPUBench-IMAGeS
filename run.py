import argparse
import os

import index

parser = argparse.ArgumentParser()
parser.add_argument('--vram', type=float, default=None,
                    help='only run benchmarks with vram usage less or equal to this value')
parser.add_argument('--stress', action='store_true',
                    help='run TensorFlow stress test')
parser.add_argument('--tensorflow', action='store_true',
                    help='run all TensorFlow flagged benchmarks')
parser.add_argument('--pytorch', action='store_true',
                    help='run all PyTorch flagged benchmarks')
parser.add_argument('--id', nargs='*', type=str,
                    help='run all specified benchmarks')
parser.add_argument('--tf-min-log-lvl', type=int, default=3,
                    help='avoid tensorflow information messages')
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_min_log_lvl)

print('GPUBench-IMAGeS')

if args.vram is None:
    max_vram = float('inf')
else:
    max_vram = args.vram

if args.stress:
    bench = index.stress()
    while True:
        print(bench.mark())

if args.tensorflow:
    for id in index.tensorflow:
        bench = index.benchmarks[id]()
        result = bench.mark()
        print(f'{id} : {result}')

if args.pytorch:
    raise NotImplementedError

if args.id is None:
    args.id = []

for id in args.id:
    if id not in index.benchmarks.keys():
        print(f'ERROR: wrong benchmark id : {id}')
    else:
        bench = index.benchmarks[id]()
        result = bench.mark()
        print(f'{id} : {result}')
