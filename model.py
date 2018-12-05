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
        p = tf.pad(act, ([3, 3], [3, 3]), "reflect")
        conv = tf.layers.conv2d(p, 3, kernel_size=[7, 7], padding="valid", strides=[1, 1],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        output = tf.nn.tanh(conv)

        return output

def discriminator(input, is_train, reuse=False):
    with tf.variable_scope("dis") as scope:
        if reuse:
            scope.reuse_variables()

        input = tf.reshape(input, shape=[-1, IMAGE_HIGHT, IMAGE_WIDTH, CHANNEL])

        # Block 1
        conv = tf.layers.conv2d(input, 64, kernel_size=[4, 4], padding="same", strides=[2, 2],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        act = tf.nn.leaky_relu(conv, alpha=0.2)

        # Block 2
        conv = tf.layers.conv2d(act, 64, kernel_size=[4, 4], padding="same", strides=[2, 2],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 3
        conv = tf.layers.conv2d(act, 128, kernel_size=[4, 4], padding="same", strides=[2, 2],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 4
        conv = tf.layers.conv2d(act, 256, kernel_size=[4, 4], padding="same", strides=[2, 2],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 5
        conv = tf.layers.conv2d(act, 512, kernel_size=[4, 4], padding="same", strides=[1, 1],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 6
        conv = tf.layers.conv2d(act, 1, kernel_size=[4, 4], padding="same", strides=[1, 1],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        f = tf.layers.flatten(conv)

        # Block 7
        d = tf.layers.dense(f, 1024, activation='tanh', kernel_initializer=tf.glorot_uniform_initializer())
        output = tf.layers.dense(f, 1, activation='sigmoid', kernel_initializer=tf.glorot_uniform_initializer())

    return output
