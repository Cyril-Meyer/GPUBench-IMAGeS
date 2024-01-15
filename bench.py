import time
import numpy as np


class Bench:
    vram = float('inf')
    id = 'root'

    def __init__(self):
        if self.id == 'root' or self.vram == float('inf'):
            raise NotImplementedError
        self.imports = dict()

    def mark(self) -> float:
        raise NotImplementedError


class Tf2(Bench):
    def __init__(self):
        global tensorflow
        import tensorflow
        global tfutils
        import tfutils

    def mark(self) -> float:
        raise NotImplementedError


class Tf2Mnist(Tf2):
    def __init__(self):
        super().__init__()

        mnist = tensorflow.keras.datasets.mnist

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        self.x_train = self.x_train[..., tensorflow.newaxis].astype('float32')
        self.x_test = self.x_test[..., tensorflow.newaxis].astype('float32')

    def mark(self) -> float:
        raise NotImplementedError


class Tf2Cifar100(Tf2):
    def __init__(self):
        super().__init__()

        cifar100 = tensorflow.keras.datasets.cifar100

        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

    def mark(self) -> float:
        raise NotImplementedError


class Tf2Mlp(Tf2Mnist):
    vram = 1.0
    id = 'TF2-MLP'

    def __init__(self):
        super().__init__()

        self.model = tensorflow.keras.models.Sequential([
                     tensorflow.keras.layers.Flatten(input_shape=(28, 28, 1)),
                     tensorflow.keras.layers.Dense(4096, activation='relu'),
                     tensorflow.keras.layers.Dense(4096, activation='relu'),
                     tensorflow.keras.layers.Dense(1024, activation='relu'),
                     tensorflow.keras.layers.Dense(10)
        ])
        self.model.compile(optimizer='adam',
                           loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    def mark(self) -> float:
        time_callback = tfutils.TimeHistory()
        history = self.model.fit(self.x_train, self.y_train,
                                 epochs=11,
                                 batch_size=256,
                                 callbacks=[time_callback],
                                 verbose=0)

        return np.mean(time_callback.times[1:])


class Tf2CnnResnet50(Tf2Cifar100):
    vram = 3.0
    id = 'TF2-CNN-ResNet50'

    def __init__(self):
        super().__init__()

        self.model = tensorflow.keras.applications.resnet50.ResNet50(
                     include_top=True,
                     weights=None,
                     input_shape=(32, 32, 3),
                     classes=100)
        self.model.compile(optimizer='adam',
                           loss=tensorflow.keras.losses.SparseCategoricalCrossentropy())

    def mark(self) -> float:
        time_callback = tfutils.TimeHistory()
        history = self.model.fit(self.x_train, self.y_train,
                                 epochs=11,
                                 batch_size=32,
                                 steps_per_epoch=128,
                                 callbacks=[time_callback],
                                 verbose=0)

        return np.mean(time_callback.times[1:])
