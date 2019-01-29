import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import scipy

from model import *
from loss import *
from PIL import Image
from cv2 import imwrite
import keras.backend as K

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 540
CHANNEL = 3
EPOCH = 1000
BATCH_SIZE = 4
TRAIN_IMAGES = 100
version = 'version' #structure
image_save_path = './step_result'
gen_loss = []
dis_loss = []

#plot loss
def plot():
    fig = plt.figure()
    plt.plot(range(EPOCH), gen_loss, label="Generator_Loss")
    plt.plot(range(EPOCH), dis_loss, label="Discriminator_Loss")

    plt.legend(loc=0, ncol=1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    '''if not os.path.exists("./plot"):
        os.makedirs("./plot")
    '''
    fig.savefig(version+".jpg")
    print("Loss function plotted")
    #plt.show()
    return


def train2():

    real_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    input_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    dis_input_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    dis_input = tf.placeholder(tf.float32, shape = [BATCH_SIZE, 1])
    target_result = tf.placeholder(tf.float32, shape = [BATCH_SIZE, 1])
    g_is_train = tf.placeholder(tf.bool)
    d_is_train = tf.placeholder(tf.bool)

    generated_image = generator(input_image, g_is_train)

    dis_result = discriminator(dis_input_image, d_is_train)

    d_loss = singleDisLoss(target_result, dis_result)

    component_loss1 = perceptual_loss(real_image, generated_image)
    component_loss2 = MSE(real_image, generated_image)
    component_loss3 = singleDisLoss(target_result, dis_input)
    g_loss = 1 * component_loss1 + 1 * component_loss2 + 1 * component_loss3

    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dis")
    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gen")
    trainer_d = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(g_loss, var_list=g_vars)

    data = load_images()
    x_train = data['A']
    y_train = data['B']
    sample = data['Sample']
    sample_result3 = sample * 127.5 + 127.5
    sampleImg3 = sample_result3[0].astype('uint8')
    imwrite(image_save_path + "/sample1.jpeg", sampleImg3)
    print("number of images: {}\nimages size and chaneel: {}\n".format(
        x_train.shape[0], x_train.shape[1:]))

    output_true_batch = np.ones((BATCH_SIZE, 1))
    output_false_batch = np.zeros((BATCH_SIZE, 1))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    print('start training...')

    for epoch in range(EPOCH):
        print("==>> Running epoch [{}/{}]...".format(epoch+1, EPOCH))
        start = time.time()
        permutated_indexes = np.random.permutation(x_train.shape[0])

        for batch in range(int(x_train.shape[0]/BATCH_SIZE)):
            batch_indexes = permutated_indexes[batch * BATCH_SIZE:(batch+1)*BATCH_SIZE]
            x_batch = x_train[batch_indexes]
            y_batch = y_train[batch_indexes]
            dl1 = 0
            dl2 = 0
            for iter in range(5):
                gen_image = sess.run(generated_image, feed_dict={input_image: x_batch, g_is_train: True})

                _, dLoss2, result = sess.run([trainer_d, d_loss, dis_result],
                    feed_dict={dis_input_image: gen_image, target_result: output_false_batch, d_is_train: True})
                dl2 = dl2 + dLoss2

            for iter in range(5):
                _, dLoss1 = sess.run([trainer_d, d_loss],
                    feed_dict={dis_input_image: y_batch, target_result: output_true_batch, d_is_train: True})
                dl1 = dl1 + dLoss1

            print("     loss True: {}, loss False: {}".format(dl1/5, dl2/5))
            dLoss = (dl1/5 + dl2/5)/2


            for iter in range(1):
                _, gLoss, pLoss, mLoss, dloss = sess.run([trainer_g, g_loss, component_loss1, component_loss2, component_loss3],
                    feed_dict={input_image: x_batch, real_image: y_batch, dis_input: result,
                    target_result: output_true_batch, g_is_train: True})

            dis_loss.append(dLoss)
            gen_loss.append(gLoss)

            print('    Batch: %d, d_loss: %f, g_loss: %f, pLoss: %f, MSE: %f, dloss: %f' % (batch+1, dLoss, gLoss, pLoss, mLoss, dloss))

        #save result per 50 epoch
        if epoch%1 == 0:
            if not os.path.exists(image_save_path):
                os.makedirs(image_save_path)

            sample_result = sess.run(generated_image, feed_dict={input_image: sample, g_is_train: True})
            sample_result = sample_result * 127.5 + 127.5
            sampleImg = sample_result[0].astype('uint8')
            imwrite(image_save_path + "/" + str(epoch) + ".jpeg", sampleImg)

            print("    Sample image saved!")
            print ('    Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    coord.request_stop()
    coord.join(threads)

def load_images():
    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, 'images/train/A')
    real_dir = os.path.join(current_dir, 'images/train/B')

    A_images = []
    index = 0
    for each in os.listdir(input_dir):
        if index >= TRAIN_IMAGES:
            break
        if each == ".DS_Store":
            continue
        img = Image.open(input_dir + '/' + each)
        resizedImg = img.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
        imgArray = np.array(resizedImg)
        imgArray = (imgArray - 127.5) / 127.5
        A_images.append(imgArray)
        index += 1

    B_images = []
    index = 0
    for each in os.listdir(real_dir):
        if index >= TRAIN_IMAGES:
            break
        if each == ".DS_Store":
            continue
        img = Image.open(real_dir + '/' + each)
        resizedImg = img.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
        imgArray = np.array(resizedImg)
        imgArray = (imgArray - 127.5) / 127.5
        B_images.append(imgArray)
        index += 1

    S_images = []
    img = Image.open("images/test/A/93.jpg")
    resizedImg = img.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
    imgArray = np.array(resizedImg)

    sampleImg2 = imgArray.astype('uint8')
    imwrite(image_save_path + "/sample0.jpeg", sampleImg2)

    imgArray = (imgArray - 127.5) / 127.5
    S_images.append(imgArray)

    S_images2 = []
    img = Image.open("images/test/B/93.jpg")
    resizedImg = img.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
    imgArray = np.array(resizedImg)

    sampleImg2 = imgArray.astype('uint8')
    imwrite(image_save_path + "/sample_result0.jpeg", sampleImg2)

    imgArray = (imgArray - 127.5) / 127.5
    S_images2.append(imgArray)

    print("Sample saved!")
    return{
        'A': np.array(A_images),
        'B': np.array(B_images),
        'Sample': np.array(S_images),
        'Sample2': np.array(S_images2)
    }


if __name__ == '__main__':
    train2()
    plot()
