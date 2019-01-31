import os
import tensorflow as tf
import numpy as np

IMAGE_HIGHT = 400
IMAGE_WIDTH = 540
CHANNEL = 3
# KI = tf.glorot_uniform_initializer()

def generator(input, is_train, reuse = False):
    with tf.variable_scope("gen", reuse = reuse):

        #print("input: "+str(input.shape))

        # Block 1
        p = tf.pad(input, ([0,0], [3, 3], [3, 3], [0,0]), "reflect")
        #print(p.shape)
        conv = tf.layers.conv2d(p, 64, kernel_size=[7, 7], padding="valid", strides=[1, 1])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)
        #print("block1: "+str(act.shape))

        # Block 1
        p = tf.pad(act, ([0,0], [3, 3], [3, 3], [0,0]), "reflect")
        #print(p.shape)
        conv = tf.layers.conv2d(p, 64, kernel_size=[7, 7], padding="valid", strides=[1, 1])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)
        res1 = act

        # Block 2
        # kernel_size=[7, 7] in dissertation
        conv = tf.layers.conv2d(act, 128, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)
        #print("block2: "+str(act.shape))

        # Block 2
        # kernel_size=[7, 7] in dissertation
        conv = tf.layers.conv2d(act, 128, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)
        res2 = act

        # Block 3
        # filters = 128, kernel_size=[7, 7] in dissertation
        conv = tf.layers.conv2d(act, 256, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)
        #print("block3: "+str(act.shape))

        # Block 3
        # filters = 128, kernel_size=[7, 7] in dissertation
        conv = tf.layers.conv2d(act, 256, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)
        res3 = act

        # 9 Res Blocks
        # kernel_size=[7, 7] in dissertation
        for i in range(9):
            temp_x = act
            conv = tf.layers.conv2d(act, 256, kernel_size=[3, 3], padding="same", strides=[1, 1])
            bn = tf.layers.batch_normalization(conv, training=is_train)
            act = tf.nn.relu(bn)
            #print("blockMid1: "+str(act.shape))
            dropout = tf.layers.dropout(act, training=is_train)
            p = tf.pad(dropout, ([0, 0], [1, 1], [1, 1], [0, 0]), "reflect")
            conv = tf.layers.conv2d(p, 256, kernel_size=[3, 3], padding="valid", strides=[1, 1])
            #bn = tf.layers.batch_normalization(conv, training=is_train)
            bn = tf.nn.relu(conv)
            #print("blockMid2: "+str(bn.shape))
            #act = tf.add(temp_x, bn)
            act = bn
            #print("block4-12: "+str(act.shape))

        act = (act + res3)/2

        # Block 13
        conv = tf.layers.conv2d_transpose(act, 128, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)
        act = (act + res2)/2
        #print(act.shape)

        # Block 13
        conv = tf.layers.conv2d_transpose(act, 128, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)

        # Block 14
        conv = tf.layers.conv2d_transpose(act, 64, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)
        #act = (act + res1)/2
        #print(act.shape)

        # Block 14
        conv = tf.layers.conv2d_transpose(act, 64, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.relu(bn)
        #act = (act + res1)/2

        # Block 15
        p = tf.pad(act, ([0, 0], [3, 3], [3, 3], [0, 0]), "reflect")
        conv = tf.layers.conv2d(p, 3, kernel_size=[7, 7], padding="valid", strides=[1, 1])
        output = tf.nn.tanh(conv)
        #print(output.shape)

    return output


def discriminator(input, is_train, reuse = False):
    with tf.variable_scope("dis") as scope:
        if reuse:
            scope.reuse_variables()

        #input = tf.reshape(input, shape=[-1, IMAGE_HIGHT, IMAGE_WIDTH, CHANNEL])

        # Block 1
        conv = tf.layers.conv2d(input, 64, kernel_size=[3, 3], padding="same", strides=[1, 1])
        act = tf.nn.leaky_relu(conv, alpha=0.2)

        # Block 2
        conv = tf.layers.conv2d(act, 64, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 3
        conv = tf.layers.conv2d(act, 128, kernel_size=[3, 3], padding="same", strides=[1, 1])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 3
        conv = tf.layers.conv2d(act, 128, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 3
        conv = tf.layers.conv2d(act, 256, kernel_size=[3, 3], padding="same", strides=[1, 1])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 3
        conv = tf.layers.conv2d(act, 256, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 4
        '''conv = tf.layers.conv2d(act, 256, kernel_size=[4, 4], padding="same", strides=[2, 2],
                                        kernel_initializer=tf.glorot_uniform_initializer())
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)
        '''

        # Block 5
        conv = tf.layers.conv2d(act, 512, kernel_size=[3, 3], padding="same", strides=[1, 1])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 5
        conv = tf.layers.conv2d(act, 512, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 6
        conv = tf.layers.conv2d(act, 1, kernel_size=[1, 1], padding="same", strides=[1, 1])
        f = tf.layers.flatten(conv)

        # Block 7
        d = tf.layers.dense(f, 1024)
        act = tf.nn.leaky_relu(d, alpha=0.2)
        output = tf.layers.dense(act, 1, activation='sigmoid')

    return output


def discriminator2(input, is_train, reuse=False):
    with tf.variable_scope("dis") as scope:
        if reuse:
            scope.reuse_variables()

        #input = tf.reshape(input, shape=[-1, IMAGE_HIGHT, IMAGE_WIDTH, CHANNEL])

        # Block 1
        conv = tf.layers.conv2d(input, 64, kernel_size=[3, 3], padding="same", strides=[1, 1])
        act = tf.nn.leaky_relu(conv, alpha=0.2)

        # Block 2
        conv = tf.layers.conv2d(act, 64, kernel_size=[3, 3], padding="same", strides=[2, 2])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 3
        conv = tf.layers.conv2d(act, 128, kernel_size=[3, 3], padding="same", strides=[1, 1])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 4
        conv = tf.layers.conv2d(act, 256, kernel_size=[3, 3], padding="same", strides=[1, 1])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 5
        conv = tf.layers.conv2d(act, 512, kernel_size=[3, 3], padding="same", strides=[1, 1])
        bn = tf.layers.batch_normalization(conv, training=is_train)
        act = tf.nn.leaky_relu(bn, alpha=0.2)

        # Block 6
        conv = tf.layers.conv2d(act, 1, kernel_size=[3, 3], padding="same", strides=[1, 1])
        f = tf.layers.flatten(conv)

        # Block 7
        d = tf.layers.dense(f, 1024)
        act = tf.nn.leaky_relu(d, alpha=0.2)
        output = tf.layers.dense(act, 1, activation='sigmoid')

    return output
