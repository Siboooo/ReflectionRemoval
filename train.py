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
    return


def data_process():
    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, 'images/train/A')
    real_dir = os.path.join(current_dir, 'images/train/B')

    #input images batch
    input_images = []
    for each in os.listdir(input_dir):
        if each == ".DS_Store":
            continue
        input_images.append (os.path.join(input_dir, each))
    all_input_images = tf.convert_to_tensor(input_images, dtype = tf.string)
    input_images_queue = tf.train.slice_input_producer([all_input_images])
    input_content = tf.read_file(input_images_queue[0])
    input_image = tf.image.decode_jpeg(input_content, channels = CHANNEL)
    sess1 = tf.Session()
    size = [IMAGE_HEIGHT, IMAGE_WIDTH]
    input_image = tf.image.resize_images(input_image, size)
    input_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    input_image = tf.cast(input_image, tf.float32)
    input_image = input_image / 255.0
    input_images_batch = tf.train.shuffle_batch([input_image],
                                        batch_size = BATCH_SIZE,
                                        num_threads = 4,
                                        capacity = 200 + 3 * BATCH_SIZE,
                                        min_after_dequeue = 200)
    input_num = len(input_images)


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

    size = [IMAGE_HEIGHT, IMAGE_WIDTH]
    real_image = tf.image.resize_images(real_image, size)
    real_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
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
    with tf.variable_scope("input"):
        real_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        input_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        squeeze_image = tf.placeholder(tf.float32, shape = [1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        is_train = tf.placeholder(tf.bool)

    generated_image = generator(input_image, is_train)

    real_result = discriminator(real_image, is_train)
    generated_result = discriminator(generated_image, is_train, reuse = True)

    d_loss = wasserstein_loss(real_result, generated_result)
    g_loss = tf.add(tf.multiply(100.0, perceptual_loss(real_image, generated_image)),
        tf.multiply(1.0, wasserstein_loss(real_image, generated_image)))

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(g_loss, var_list=g_vars)

    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    real_image_batch, input_image_batch, input_num, real_number = data_process()
    sample_data = sample_process()
    squeezed_image = tf.squeeze(squeeze_image)
    sample_image = tf.cast(squeezed_image, tf.float32)
    sample_image = tf.multiply(255.0, sample_image)
    squeezed_image2 = tf.cast(sample_image, tf.uint8)
    encoded_image = tf.image.encode_jpeg(squeezed_image2, "rgb")

    batch_num = int(input_num / BATCH_SIZE)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    print('total training sample num: %d' % input_num)
    print('total training traget num: %d' % real_number)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (BATCH_SIZE, batch_num, EPOCH))
    print('start training...')

    #restore the variables from same version
    if os.path.exists("./vars/"+version+".ckpt"):
        print("Reload vars")
        saver.restore(sess, "./vars/"+version+".ckpt")

    for epoch in range(EPOCH):
        print("==>> Running epoch [{}/{}]...".format(epoch+1, EPOCH))

        for batch in range(batch_num):
            # train descriminator
            inputs, targets = sess.run([real_image_batch, input_image_batch])
            for iter in range(5):
                sess.run(d_clip)

                _, dLoss = sess.run([trainer_d, d_loss],
                    feed_dict={input_image: inputs, real_image: targets, is_train: True})

            # train generator
            for iter in range(1):
                _, gLoss = sess.run([trainer_g, g_loss],
                    feed_dict={input_image: inputs, real_image: targets, is_train: True})

            print('    Batch: %d, d_loss: %f, g_loss: %f' % (batch+1, dLoss, gLoss))

            dis_loss[epoch] = dLoss
            gen_loss[epoch] = gLoss

        #save variables per 100 epoch
        if epoch%100 == 0:
            if not os.path.exists("./vars"):
                os.makedirs("./vars")
            save_path = saver.save(sess, "./vars/"+version+".ckpt")
            print("Variables saved in path: %s" % save_path)

        #save result per 50 epoch
        if epoch%50 == 0:
            if not os.path.exists(image_save_path):
                os.makedirs(image_save_path)
            sample_input = sess.run(sample_data)
            sample_result = sess.run(generated_image, feed_dict={input_image: sample_input, is_train: False})
            sample = sess.run(encoded_image, feed_dict={squeeze_image: sample_result})
            file = tf.write_file(image_save_path + "/" + str(epoch) + ".jpeg", sample)
            sess.run(file)
            print("Sample image saved!")

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    train()
    plot()
