import numpy as np
import bench

tensorflow = np.array([
    ('TF2-MLP', bench.Tf2Mlp),
    ('TF2-CNN-ResNet50', bench.Tf2CnnResnet50)
])

'''
pytorch = np.array([

])
misc = np.array([

])
'''
# assert len(list(set(tensorflow[:, 0]) & set(pytorch[:, 0]) & set(misc[:, 0]))) == 0

tensorflow = dict(tensorflow)
benchmarks = tensorflow