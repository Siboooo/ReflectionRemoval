import os
import tensorflow as tf
import numpy as np
import scipy.misc

IMAGE_HIGHT = 400
IMAGE_WIDTH = 540
CHANNEL = 3

def generator(input, is_train, reuse = False):

    with tf.variable_scope("gen") as scope:
        if reuse:
            scope.reuse_variables()

        input = tf.reshape(input, shape=[IMAGE_HIGHT, IMAGE_WIDTH, CHANNEL])

        # Block 1
        # ReflectionPadding2D((3, 3))(inputs)
        p = tf.pad(input, ([3, 3], [3, 3]), "reflect")
        conv = tf.layers.conv2d(p, 64, kernel_size=[7, 7], padding="valid", strides=[1, 1],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)

        # Block 2
        # kernel_size=[7, 7] in dissertation
        conv = tf.layers.conv2d(act, 128, kernel_size=[3, 3], padding="same", strides=[2, 2],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)

        # Block 3
        # filters = 128, kernel_size=[7, 7] in dissertation
        conv = tf.layers.conv2d(act, 256, kernel_size=[3, 3], padding="same", strides=[2, 2],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)

        # 9 Res Blocks
        # kernel_size=[7, 7] in dissertation
        for i in range(9):
            temp_x = act
            conv = tf.layers.conv2d(act, 256, kernel_size=[3, 3], padding="valid", strides=[1, 1],
                                            kernel_initializer=tf.glorot_uniform_initializer())
            bn = tf.layers.batch_normalization(conv, training=is_train)
            act = tf.nn.relu(bn)
            dropout = tf.layer.dropout(act, training=is_train)
            # ReflectionPadding2D(1, 1)(x)
            p = tf.pad(dropout, ([1, 1], [1, 1]), "reflect")
            conv = tf.layers.conv2d(p, 256, kernel_size=[3, 3], padding="valid", strides=[1, 1],
                                            kernel_initializer=tf.glorot_uniform_initializer())
            bn = tf.layers.batch_normalization(conv, training=is_train)
            act = tf.add(temp_x, bn)

        # Block 13
        conv = tf.layers.conv2d_transpose(act, 128, kernel_size=[3, 3], padding="same",
                                strides=[2, 2], kernel_initializer=tf.glorot_uniform_initializer())
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)

        # Block 14
        conv = tf.layers.conv2d_transpose(act, 64, kernel_size=[3, 3], padding="same",
                                strides=[2, 2], kernel_initializer=tf.glorot_uniform_initializer())
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)

        # Block 15
        # ReflectionPadding2D((3, 3))(inputs)
        p = tf.pad(act, ([3, 3], [3, 3]), "reflect")
        conv = tf.layers.conv2d(p, 3, kernel_size=[7, 7], padding="valid", strides=[1, 1],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        output = tf.nn.tanh(conv)

        # might cause the unsatisfied result
        output = tf.add(output, input)
        output = tf.truediv(output, 2)

        return output
