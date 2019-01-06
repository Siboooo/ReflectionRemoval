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
TRAIN_IMAGES = 400
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
    if not os.path.exists("./plot"):
        os.makedirs("./plot")
    fig.savefig("./plot/"+version+".jpg")
    print("Loss function plotted")
    #plt.show()
    return


def train2():

    '''
    with tf.variable_scope("input"):
        real_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        input_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        dis_input_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        target_result = tf.placeholder(tf.float32, shape = [BATCH_SIZE, 1])
        g_is_train = tf.placeholder(tf.bool)
        d_is_train = tf.placeholder(tf.bool)
    '''

    #'''
    real_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    input_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    dis_input_image = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    target_result = tf.placeholder(tf.float32, shape = [BATCH_SIZE, 1])
    g_is_train = tf.placeholder(tf.bool)
    d_is_train = tf.placeholder(tf.bool)
    #'''

    generated_image = generator(input_image, g_is_train)

    dis_result = discriminator(dis_input_image, d_is_train)
    real_result = discriminator(real_image, d_is_train, reuse = True)
    #dis_result = discriminator(dis_input_image, d_is_train)
    #generated_result = discriminator(generated_image, d_is_train, reuse = True)

    #d_loss = wasserstein_loss(target_result, dis_result)
    #d_loss = discriminator_loss(real_result, dis_result)
    d_loss = dice_coefficient(target_result, dis_result)

    g_loss = perceptual_loss(real_image, generated_image)
    #g_loss = tf.add(tf.multiply(1000.0, pLoss), tf.multiply(1.0, wasserstein_loss(real_image, generated_image)))
    #g_loss = tf.add(tf.multiply(100.0, perceptual_loss(real_image, generated_image)),
    #    tf.multiply(1.0, wasserstein_loss(real_image, generated_image)))

    #t_vars = tf.trainable_variables()
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dis")
    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gen")
    #d_vars = [var for var in t_vars if 'dis' in var.name]
    #g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(g_loss, var_list=g_vars)

    data = load_images()
    x_train = data['A']
    y_train = data['B']
    sample = data['Sample']
    sample_result3 = sample[0] * 127.5 + 127.5
    sampleImg3 = sample_result3.astype('uint8')
    imwrite(image_save_path + "/sample1.jpeg", sampleImg3)
    print("number of images: {}\nimages size and chaneel: {}\n".format(
        x_train.shape[0], x_train.shape[1:]))

    output_true_batch = np.ones((BATCH_SIZE, 1))
    output_false_batch = np.zeros((BATCH_SIZE, 1))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    #sess = tf.Session()
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

            for iter in range(5):

                gen_image = sess.run(generated_image, feed_dict={input_image: x_batch, g_is_train: False})

                _, dLoss2 = sess.run([trainer_d, d_loss],
                    feed_dict={dis_input_image: gen_image, target_result: output_false_batch, d_is_train: True})

                '''
                disResult2 = sess.run(dis_result, feed_dict={dis_input_image: x_batch, d_is_train: False})
                print(disResult2)
                print(output_false_batch)
                print("falseResult: "+str(output_false_batch - disResult2))
                print(sess.run(wasserstein_loss(output_false_batch, disResult2)))
                print(sess.run(d_loss, feed_dict={dis_input_image: gen_image, target_result: output_false_batch, d_is_train: False}))
                print("dLoss2: "+str(dLoss2))
                print(" ")
                '''

                dLoss1 = 0
                #_, dLoss1 = sess.run([trainer_d, d_loss],
                #    feed_dict={dis_input_image: y_batch, target_result: output_true_batch, d_is_train: True})

                '''disResult = sess.run(dis_result, feed_dict={dis_input_image: y_batch, d_is_train: False})
                print(disResult)
                print(output_true_batch)
                print("trueResult: "+str(output_true_batch - disResult))
                print(sess.run(d_loss, feed_dict={dis_input_image: y_batch, target_result: output_true_batch, d_is_train: False}))
                print("dLoss1: "+str(dLoss1))
                print(" ")
                '''
                '''
                _, dLoss1 = sess.run([trainer_d, d_loss],
                    feed_dict={dis_input_image: y_batch, target_result: output_true_batch, d_is_train: False})

                disResult = sess.run(dis_result, feed_dict={dis_input_image: y_batch, d_is_train: False})
                print(disResult)
                print(output_true_batch)
                print("result: "+str(output_true_batch - disResult))

                gen_image = sess.run(generated_image, feed_dict={input_image: x_batch, g_is_train: False})

                _, dLoss2 = sess.run([trainer_d, d_loss],
                    feed_dict={dis_input_image: gen_image, target_result: output_false_batch, d_is_train: False})

                disResult2 = sess.run(dis_result, feed_dict={dis_input_image: x_batch, d_is_train: False})
                print(disResult2)
                print(output_false_batch)
                print("result: "+str(output_false_batch - disResult2))
                '''

                '''_, dLoss = sess.run([trainer_d, d_loss],
                    feed_dict={input_image: x_batch, real_image: y_batch, d_is_train: True, g_is_train: False})
                dis_loss.append(dLoss)
                '''
                print("     loss True: {}, loss False: {}".format(dLoss1, dLoss2))
                dLoss = (dLoss1 + dLoss2)/2


            for iter in range(1):
                #genImage = sess.run(generated_image, feed_dict={input_image: x_batch, g_is_train: False})
                sImg, gLoss, _ = sess.run([generated_image, g_loss, trainer_g],
                    feed_dict={input_image: x_batch, real_image: y_batch, g_is_train: True})
                sImg = sImg * 127.5 + 127.5
                sampleImg = sImg[0].astype('uint8')
                imwrite(image_save_path + "/gen" + str(epoch) + ".jpeg", sampleImg)
                dis_loss.append(dLoss)
                gen_loss.append(gLoss)

            print('    Batch: %d, d_loss: %f, g_loss: %f' % (batch+1, dLoss, gLoss))

        '''
        #save variables per 100 epoch
        if epoch%100 == 0:
            if not os.path.exists("./vars"):
                os.makedirs("./vars")
            save_path = saver.save(sess, "./vars/"+version+".ckpt")
            print("Variables saved in path: %s" % save_path)
        '''

        #save result per 50 epoch
        if epoch%1 == 0:
            if not os.path.exists(image_save_path):
                os.makedirs(image_save_path)

            sample_result = sess.run(generated_sample, feed_dict={input_image: sample})
            sample_result = sample_result * 127.5 + 127.5
            sampleImg = sample_result[0].astype('uint8')
            imwrite(image_save_path + "/" + str(epoch) + ".jpeg", sampleImg)

            #sampleImg = Image.fromarray(sample_result)
            #im.save(image_save_path + "/" + str(epoch) + ".jpeg")
            #file = tf.write_file(image_save_path + "/" + str(epoch) + ".jpeg", sample_result.astype('uint8'))
            #sess.run(file)
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


    return{
        'A': np.array(A_images),
        'B': np.array(B_images),
        'Sample': np.array(S_images),
        'Sample2': np.array(S_images2)
    }

if __name__ == '__main__':
    train2()
    plot()
