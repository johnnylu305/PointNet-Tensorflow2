import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Sequential, activations, initializers, Input, optimizers


class Transform_Net(layers.Layer):
    def __init__(self, name, K):
        super(Transform_Net, self).__init__()
        # conv2d: [filters, kernel_size, stride, padding, activation, batch_norm]
        # dense: [neurons, activation, batch_norm, drop_out]
        # trans: [matrix_size]
        net_para = []
        if 'Tnet_1' in name:
            net_para = [['{}_reshape1'.format(name), [-1, 3, 1]],
                        ['{}_conv2d_1'.format(name), [64, (1, 3), (1, 1), 'valid', 'relu', True]],
                        ['{}_conv2d_2'.format(name), [128, (1, 1), (1, 1), 'valid', 'relu', True]],
                        ['{}_conv2d_3'.format(name), [1024, (1, 1), (1, 1), 'valid', 'relu', True]],
                        ['{}_pooling1'.format(name), ['GlobalMaxPooling2D']],
                        ['{}_reshape2'.format(name), [-1]],
                        ['{}_dense_1'.format(name), [512, 'relu', True, None]],
                        ['{}_dense_2'.format(name), [256, 'relu', True, None]],
                        ['{}_matrix_1'.format(name), [K]],
                        ['{}_reshape3'.format(name), [K, K]]]
        elif 'Tnet_2' in name:
            net_para = [['{}_conv2d_1'.format(name), [64, (1, 1), (1, 1), 'valid', 'relu', True]],
                        ['{}_conv2d_2'.format(name), [128, (1, 1), (1, 1), 'valid', 'relu', True]],
                        ['{}_conv2d_3'.format(name), [1024, (1, 1), (1, 1), 'valid', 'relu', True]],
                        ['{}_pooling1'.format(name), ['GlobalMaxPooling2D']],
                        ['{}_reshape2'.format(name), [-1]],
                        ['{}_dense_1'.format(name), [512, 'relu', True, None]],
                        ['{}_dense_2'.format(name), [256, 'relu', True, None]],
                        ['{}_matrix_1'.format(name), [K]],
                        ['{}_reshape3'.format(name), [K, K]]]
        self.model = Sequential()
        for layer in net_para:
            if 'conv2d' in layer[0]:
                self.model.add(conv2d(layer))
            elif 'dense' in layer[0]:
                self.model.add(dense(layer))
            elif 'matrix' in layer[0]:
                self.model.add(matrix_multi(layer))
            elif 'reshape' in layer[0]:
                self.model.add(reshape(layer))
            elif 'pooling' in layer[0]:
                self.model.add(pooling(layer))

    def call(self, x):
        x = self.model(x)
        return x


def set_bn(net, momentum):
    if isinstance(net, layers.BatchNormalization):
        net.momentum = momentum
    elif isinstance(net, Transform_Net) or isinstance(net, conv2d) or isinstance(net, dense):
        for subnet in net.model.layers:
            set_bn(subnet, momentum)


class lr_decay(optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, decay_steps=6250, decay_rate=0.7, staircase=True, clip=0.00001):
        super(lr_decay, self).__init__()
        self.clip = clip
        self.schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr,
                                                                       decay_steps=decay_steps,
                                                                       decay_rate=decay_rate,
                                                                       staircase=staircase)
    def __call__(self, step):
        return tf.maximum(self.schedule(step), self.clip)


class bn_decay(optimizers.schedules.LearningRateSchedule):
    def __init__(self, momentum=0.5, decay_steps=6250, decay_rate=0.5, staircase=True, clip=0.99):
        super(bn_decay, self).__init__()
        self.clip = clip
        self.schedule = tf.keras.optimizers.schedules.ExponentialDecay(1-momentum,
                                                                       decay_steps=decay_steps,
                                                                       decay_rate=decay_rate,
                                                                       staircase=staircase)
    def __call__(self, step):
        return tf.minimum(1-self.schedule(step), self.clip)


class conv2d(layers.Layer):
    def __init__(self, layer):
        super(conv2d, self).__init__()
        # work like a shared weight mlp for each point
        self.model = Sequential()
        self.model.add(layers.Conv2D(filters=layer[1][0],
                                     kernel_size=layer[1][1],
                                     strides=layer[1][2],
                                     padding=layer[1][3]))
        # batch norm
        if layer[1][5]:
            self.model.add(layers.BatchNormalization(momentum=0.5))
        # activation
        if layer[1][4]=='relu':
            self.model.add(layers.Activation(activations.relu))

    def call(self, x):
        x = self.model(x)
        return x


class dense(layers.Layer):
    def __init__(self, layer):
        super(dense, self).__init__()
        self.model = Sequential()
        self.dropout = None
        self.model.add(layers.Dense(layer[1][0]))
        # batch norm
        if layer[1][2]:
            self.model.add(layers.BatchNormalization(momentum=0.5))
        # activation
        if layer[1][1]=='relu':
            self.model.add(layers.Activation(activations.relu))
        if layer[1][3]:
            self.model.add(layers.Dropout(layer[1][3]))

    def call(self, x):
        x = self.model(x)
        return x


class matrix_multi(layers.Layer):
    def __init__(self, layer):
        super(matrix_multi, self).__init__()
        self.model = Sequential()
        self.model.add(layers.Dense(units=layer[1][0]**2,
                                    kernel_initializer=initializers.Zeros(),
                                    bias_initializer=eye_init))

    def call(self, x):
        x = self.model(x)
        x = x
        return x


class reshape(layers.Layer):
    def __init__(self, layer):
        super(reshape, self).__init__()
        self.model = Sequential()
        self.model.add(layers.Reshape(target_shape=layer[1]))

    def call(self, x):
        x = self.model(x)
        return x


class pooling(layers.Layer):
    def __init__(self, layer):
        super(pooling, self).__init__()
        self.model = Sequential()
        if layer[1][0]=='GlobalMaxPooling2D':
            self.model.add(layers.GlobalMaxPooling2D())       

    def call(self, x):
        x = self.model(x)
        return x


def eye_init(shape, dtype=tf.float32):
    assert len(shape)==1
    return tf.Variable(np.eye(int(shape[0]**0.5)).flatten(), dtype=dtype)
