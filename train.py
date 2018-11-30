import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy

from model import *
from loss import *

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 540
CHANNEL = 3
EPOCH = 2
BATCH_SIZE = 1
version = 'version' #structure
image_save_path = './step_result'
gen_loss = np.zeros([EPOCH])
dis_loss = np.zeros([EPOCH])


#plot loss
def plot():
    fig = plt.figure()
    plt.plot(range(EPOCH), gen_loss, label="Generator_Loss")
    plt.plot(range(EPOCH), dis_loss, label="Discriminator_Loss")

    plt.legend(loc=0, ncol=1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if not os.path.exists("./plot"):
        os.makedirs("./plot")
    fig.savefig("./plot/"+version+".jpg")
    print("Loss function plotted")
    #plt.show()
    return


def data_process():
    #print("Data process 1")
    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, 'images/train/A')
    real_dir = os.path.join(current_dir, 'images/train/B')

    #print("Data process 2")
    #input images batch
    input_images = []
    #print("Data process 2.1")
    for each in os.listdir(input_dir):
        if each == ".DS_Store":
            continue
        input_images.append (os.path.join(input_dir, each))
    #print("Data process 2.2")
    all_input_images = tf.convert_to_tensor(input_images, dtype = tf.string)
    input_images_queue = tf.train.slice_input_producer([all_input_images])
    input_content = tf.read_file(input_images_queue[0])
    input_image = tf.image.decode_jpeg(input_content, channels = CHANNEL)
    #print("Data process 2.3")
    sess1 = tf.Session()
    #print (sess1.run(input_image))
    #print (input_image.get_shape())
    size = [IMAGE_HEIGHT, IMAGE_WIDTH]
    input_image = tf.image.resize_images(input_image, size)
    input_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    #print (input_image.get_shape())
    input_image = tf.cast(input_image, tf.float32)
    input_image = input_image / 255.0
    input_images_batch = tf.train.shuffle_batch([input_image],
                                        batch_size = BATCH_SIZE,
                                        num_threads = 4,
                                        capacity = 200 + 3 * BATCH_SIZE,
                                        min_after_dequeue = 200)
    input_num = len(input_images)


    #print("Data process 3")
    #real images batch
    real_images = []
    for each in os.listdir(real_dir):
        if each == ".DS_Store":
            continue
        real_images.append (os.path.join(real_dir, each))
    all_real_images = tf.convert_to_tensor(real_images, dtype = tf.string)
    real_images_queue = tf.train.slice_input_producer([all_real_images])
    real_content = tf.read_file(real_images_queue[0])
    real_image = tf.image.decode_jpeg(real_content, channels = CHANNEL)
    sess1 = tf.Session()
    #print (sess1.run(real_image))
    #print (real_image.get_shape())
    size = [IMAGE_HEIGHT, IMAGE_WIDTH]
    real_image = tf.image.resize_images(real_image, size)
    real_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    #print (real_image.get_shape())
    real_image = tf.cast(real_image, tf.float32)
    real_image = real_image / 255.0
    real_images_batch = tf.train.shuffle_batch([real_image],
                                        batch_size = BATCH_SIZE,
                                        num_threads = 4,
                                        capacity = 200 + 3 * BATCH_SIZE,
                                        min_after_dequeue = 200)
    real_num = len(real_images)

    return real_images_batch, input_images_batch, input_num, real_num


def sample_process():
    sample_image_content = tf.gfile.FastGFile("images/test/A/93.jpg", 'rb').read()
    sample_image = tf.image.decode_jpeg(sample_image_content, channels = CHANNEL)
    '''sess1 = tf.Session()
    #print (sess1.run(sample_image))
    print (sample_image.get_shape())
    size = [IMAGE_HEIGHT, IMAGE_WIDTH]
    sample_image = tf.image.resize_images(sample_image, size)
    sample_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    print (sample_image.get_shape())'''
    sample_image = tf.cast(sample_image, tf.float32)
    sample_image = sample_image / 255.0
    sample_image = tf.expand_dims(sample_image, 0)

    return sample_image

def train():
    #print("check point 1")
    with tf.variable_scope("input"):
        real_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        input_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        squeeze_image = tf.placeholder(tf.float32, shape = [1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        is_train = tf.placeholder(tf.bool)

    generated_image = generator(input_image, is_train)

    real_result = discriminator(real_image, is_train)
    generated_result = discriminator(generated_image, is_train, reuse = True)

    #print("check point 2")
    d_loss = wasserstein_loss(real_result, generated_result)
    g_loss = tf.add(tf.multiply(100.0, perceptual_loss(real_image, generated_image)),
        tf.multiply(1.0, wasserstein_loss(real_image, generated_image)))

    #print("check point 3")
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(g_loss, var_list=g_vars)

    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    #print("check point 4")
    real_image_batch, input_image_batch, input_num, real_number = data_process()
    sample_data = sample_process()
    squeezed_image = tf.squeeze(squeeze_image)
    sample_image = tf.cast(squeezed_image, tf.float32)
    sample_image = tf.multiply(255.0, sample_image)
    squeezed_image2 = tf.cast(sample_image, tf.uint8)
    encoded_image = tf.image.encode_jpeg(squeezed_image2, "rgb")

    #print("check point 5")
    batch_num = int(input_num / BATCH_SIZE)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()

    #print("check point 6")
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    #print("check point 7")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
