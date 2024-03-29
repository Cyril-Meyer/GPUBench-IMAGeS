# GPUBench-IMAGeS
IMAGeS team GPU Benchmark for Deep Learning

# Usage
```
usage: run.py [-h] [--vram VRAM] [--tensorflow] [--pytorch]
              [--id [ID [ID ...]]] [--tf-min-log-lvl TF_MIN_LOG_LVL]

optional arguments:
  --vram VRAM           only run benchmarks with vram usage less or equal to
                        this value
  --stress              run TensorFlow stress test
  --tensorflow          run all TensorFlow flagged benchmarks
  --pytorch             run all PyTorch flagged benchmarks
  --id [ID [ID ...]]    run all specified benchmarks
  --tf-min-log-lvl TF_MIN_LOG_LVL
                        avoid tensorflow information messages
```

### Examples
* run TensorFlow benchmarks `python run.py --tensorflow`
* run TensorFlow stress test `python run.py --stress`

# Benchmarks reference values
Values represent mean duration per epoch of the test.
Lower score are better.

## TensorFlow
* Dell Precision 5820 (002KVM) 64G (Python 3.6, TensorFlow 2.6.2)
  * GPU RTX 2080 TI
  * CPU W-2135
* Dell Precision 5820 (06JWJY) 32G (Python 3.10, TensorFlow 2.13.0 (α), 2.16 (β) and docker with 2.15 (γ))
  * GPU RTX 4090
  * CPU W-2255

| Benchmark         | 2080 TI | 4090 α  | 4090 β  | 4090 γ  | W-2135  | W-2255  | 2080S |
|-------------------|--------:|--------:|--------:|--------:|--------:|--------:|------:|
| TF2-MLP           |     1.2 |     1.0 |     0.8 |     1.1 |    23.6 |    28.4 |   1.7 |
| TF2-CNN-ResNet50  |     5.2 |     4.8 |     2.5 |     5.3 |   111.2 |    16.7 |   4.9 |
| TF2-CNN-ResNet101 |     8.8 |     7.4 |     4.3 |     9.0 |   123.5 |    28.8 |   9.1 |

# Stress test
Stress test a.k.a GPGPU as winter heating are infinite test that allow you to
test the stability of your setup.
Values must be stable over time.

Reference values are as following:

| Stress test       | Time (s) |
|-------------------|---------:|
| RTX 4090          |       39 |
| RTX 2080 TI       |       67 |
| W-2135            |     1442 |
| W-2255            |     2725 |

# Tips and tricks
To check that your GPU is correctly installed, you can run the benchmark without
GPU to check that the score difference is indeed present.

```
CUDA_VISIBLE_DEVICES=-1 python run.py [...]
```
