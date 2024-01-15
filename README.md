# GPUBench-IMAGeS
IMAGeS team GPU Benchmark for Deep Learning

# Usage
```
usage: run.py [-h] [--vram VRAM] [--tensorflow] [--pytorch]
              [--id [ID [ID ...]]] [--tf-min-log-lvl TF_MIN_LOG_LVL]

optional arguments:
  --vram VRAM           only run benchmarks with vram usage less or equal to
                        this value
  --tensorflow          run all TensorFlow flagged benchmarks
  --pytorch             run all PyTorch flagged benchmarks
  --id [ID [ID ...]]    run all specified benchmarks
  --tf-min-log-lvl TF_MIN_LOG_LVL
                        avoid tensorflow information messages
```

### Examples
* run TensorFlow benchmarks `python run.py --tensorflow`

# Reference values

## TensorFlow
* Dell Precision 5820 (002KVM) 64G
  * GPU RTX 2080 TI
  * CPU W-2135
* Dell Precision 5820 (06JWJY) 32G
  * GPU RTX 4090
  * CPU W-2255

| Benchmark         | 2080 TI | W-2135  | W-2255  | 4090    |
|-------------------|--------:|--------:|--------:|--------:|
| TF2-MLP           |     1.2 |    23.6 |    28.4 |         |
| TF2-CNN-ResNet50  |     5.2 |   111.2 |    16.7 |         |
| TF2-CNN-ResNet101 |     8.8 |   123.5 |    28.8 |         |

# Tips and tricks
To check that your GPU is correctly installed, you can run the benchmark without
GPU to check that the score difference is indeed present.

```
CUDA_VISIBLE_DEVICES=-1 python run.py [...]
```
