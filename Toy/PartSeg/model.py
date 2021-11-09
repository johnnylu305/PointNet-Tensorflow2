import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses, activations
from module import Transform_Net, conv2d, reshape, dense, pooling, lr_decay, set_bn, bn_decay, concate


class Part_Segmentation(layers.Layer):
    def __init__(self, name, lr, num_point):
        super(Part_Segmentation, self).__init__()
        
        # define hyperparameter
        lr_schedule = lr_decay(lr,
                               decay_steps=10550,
                               decay_rate=0.5,
                               staircase=True,
                               clip=0.00001)
        self.bn_decay = bn_decay(momentum=0.5, 
                                 decay_steps=10550*2, 
                                 decay_rate=0.5, 
                                 staircase=True, 
                                 clip=0.99)
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # define neural network
        # Tnet: [K]
        # conv2d: [filters, kernel, stride, padding, activation, batch norm]
        # dense: [neurons, activation, batch norm, drop out]
        n = "Classification"
        self.net_para = [['{}_Tnet_1'.format(n), [3]],
                    ['{}_matrix_1'.format(n), []],
                    ['{}_reshape_1'.format(n), [-1, 3, 1]],
                    ['{}_conv2d_1'.format(n), [64, (1, 3), (1, 1), 'valid', 'relu', True, None]],
                    ['{}_conv2d_2'.format(n), [128, (1, 1), (1, 1), 'valid', 'relu', True, None]],
                    ['{}_conv2d_3'.format(n), [128, (1, 1), (1, 1), 'valid', 'relu', True, None]],
                    ['{}_Tnet_2'.format(n), [128]],
                    ['{}_reshape_2'.format(n), [-1, 128]],
                    ['{}_matrix_2'.format(n), []],
                    ['{}_reshape_3'.format(n), [-1, 1, 128]],
                    ['{}_conv2d_4'.format(n), [512, (1, 1), (1, 1), 'valid', 'relu', True, None]],
                    ['{}_conv2d_5'.format(n), [2048, (1, 1), (1, 1), 'valid', 'relu', True, None]],
                    ['{}_pooling_1'.format(n), ['GlobalMaxPooling2D']],
                    #['{}_reshape_3'.format(n), [-1]], 
                    #['{}_dense_1'.format(n), [256, 'relu', True, None]],
                    #['{}_dense_2'.format(n), [256, 'relu', True, 0.3]],
                    #['{}_dense_3'.format(n), [16, None, None, None]],
                    #['{}_softmax'.format(n), ['softmax']]
                    ]

        self.seg_para = [['{}_Concate_1'.format(name), [[1, 1, 1], 3, [1, num_point, 1, 1], 3]],
                         ['{}_conv2d_1'.format(name), [256, (1, 1), (1, 1), 'valid', 'relu', True, 0.2]],
                         ['{}_conv2d_2'.format(name), [256, (1, 1), (1, 1), 'valid', 'relu', True, 0.2]],
                         ['{}_conv2d_3'.format(name), [128, (1, 1), (1, 1), 'valid', 'relu', True, None]],
                         ['{}_conv2d_4'.format(name), [2, (1, 1), (1, 1), 'valid', None, None, None]],
                         ['{}_reshape3'.format(name), [-1, 2]], 
                         ['{}_softmax'.format(name), ['softmax']]]

        self.nets = []
        for layer in self.net_para:
            if "Tnet" in layer[0]:
                self.nets.append(Transform_Net(name=layer[0], K=layer[1][0]))
            elif 'conv2d' in layer[0]:
                self.nets.append(conv2d(layer))
            elif 'dense' in layer[0]:
                self.nets.append(dense(layer))
            elif 'reshape' in layer[0]:
                self.nets.append(reshape(layer))
            elif 'pooling' in layer[0]:
                self.nets.append(pooling(layer))
            elif 'matrix' in layer[0]:
                self.nets.append(layers.Dot(axes=(-1, -1)))
            elif 'softmax' in layer[0]:
                self.nets.append(layers.Activation(activations.softmax))

        self.seg_nets = []
        for layer in self.seg_para:
            if 'conv2d' in layer[0]:
                self.seg_nets.append(conv2d(layer))
            elif 'Concate' in layer[0]:
                self.seg_nets.append(concate(layer))
            elif 'reshape' in layer[0]:
                self.seg_nets.append(reshape(layer))
            elif 'pooling' in layer[0]:
                self.seg_nets.append(pooling(layer))
            elif 'softmax' in layer[0]:
                self.seg_nets.append(layers.Activation(activations.softmax))

    def call(self, x, cls_label):
        trans_matrix = None
        momentum = self.bn_decay(self.opt.iterations)
        layers_out = []
        max_put = None
        for i, layer in enumerate(self.net_para):
            if "Tnet" in layer[0]:
                trans_matrix = self.nets[i](x)
                set_bn(self.nets[i], momentum)
            elif "matrix" in layer[0]:
                x = self.nets[i]([x, trans_matrix])
            elif "dense" in layer[0]:
                x = self.nets[i](x)
                set_bn(self.nets[i], momentum)
            elif "conv2d" in layer[0]:
                x = self.nets[i](x)
                layers_out.append(x)
                set_bn(self.nets[i], momentum)
            elif "pooling" in layer[0]:
                x = self.nets[i](x)
                max_out = x
            else:
                x = self.nets[i](x)

        for i, layer in enumerate(self.seg_para):
            if "Concate" in layer[0]:
               x = self.seg_nets[i](tf.one_hot(cls_label, 1), layers_out, max_out)
            elif "conv2d" in layer[0]:
                x = self.seg_nets[i](x)
                layers_out.append(x)
                set_bn(self.nets[i], momentum)
            else:
                x = self.seg_nets[i](x)
        return x, trans_matrix

    def loss(self, pred, seg_label, matrix, alpha): 
        K = matrix.shape[1]
        mat = tf.matmul(matrix, tf.transpose(matrix, perm=[0,2,1]))
        # orthogonal matrix cosntraint
        l2_norm = tf.nn.l2_loss(mat-tf.constant(np.eye(K), dtype=tf.float32))
        seg_loss = tf.reduce_mean(losses.SparseCategoricalCrossentropy(from_logits=False)(seg_label, pred))
        return seg_loss+alpha*l2_norm

    def accuracy(self, pred, label):
        correct_predictions = tf.equal(pred, label)
        acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return acc
