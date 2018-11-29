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
