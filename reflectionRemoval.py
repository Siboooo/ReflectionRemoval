from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import skimage
import glob
import imageio
import shutil
import os
import time
from skimage import io, filters
from skimage.color import rgb2gray
from PIL import Image
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Lambda, Concatenate
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16

image_save_path = './step_result'
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
CHANNEL = 3
BATCH_SIZE = 10
#Set the upper limit of the TRAIN_IMAGES
TRAIN_IMAGES = 2000
EPOCH = 1000

def train():
    print("Loading Dataset")
    data = load_images()
    print("Data loaded")
    #split data into several parts
    x_train = data['A']
    x_train = x_train.astype('float32')
    y_train = data['B']
    y_train = y_train.astype('float32')
    sample = data['Sample']
    sample = sample.astype('float32')
    sample2 = data['Sample2']
    sample2 = sample2.astype('float32')

    reflection = (sample - sample2)/2
    reflection = batch_rgb2gray_Contrast(reflection)
    reflection = np.squeeze(reflection)

    if not os.path.exists("./models"):
        os.makedirs("./models")

    #process data using the Sobel operator
    x_edge_dataset = batch_sobel(x_train)
    y_edge_dataset = batch_sobel(y_train)

    try:
        #edgeMap = load_model("./models/edgeMap.h5")
        json_file = open('./models/edgeMap.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        edgeMap = tf.keras.models.model_from_json(loaded_model_json)
        edgeMap.load_weights("./models/edgeMap.h5")
        print("Load edgeMap from disk")
        edgeMap.compile(loss=edgeMSE, optimizer=tf.keras.optimizers.Adam(1e-4))
        #edgeMap.fit(x_edge_dataset, y_edge_dataset, batch_size=BATCH_SIZE, epochs=500, verbose=1)

        #edgeMap.save("./models/edgeMap.h5")
        model_json = edgeMap.to_json()
        with open("./models/edgeMap.json", "w") as json_file:
            json_file.write(model_json)
        edgeMap.save_weights("./models/edgeMap.h5")
        print("Saved edgeMap to disk")

    except:
        print("Train new edgeMap")
        edgeMap = edge_model()
        edgeMap.compile(loss=edgeMSE, optimizer=tf.keras.optimizers.Adam(1e-4))
        edgeMap.fit(x_edge_dataset, y_edge_dataset, batch_size=BATCH_SIZE, epochs=1000, verbose=1)

        #edgeMap.save("./models/edgeMap.h5")
        model_json = edgeMap.to_json()
        with open("./models/edgeMap.json", "w") as json_file:
            json_file.write(model_json)
        edgeMap.save_weights("./models/edgeMap.h5")
        print("Saved new edgeMap to disk")

    im = batch_sobel(sample)
    samplemap = edgeMap.predict_on_batch(im)
    image = np.squeeze(samplemap)
    for i in range(image.shape[0]):
        io.imsave(image_save_path + "/samplemap"+ str(i) + ".jpeg", image[i].astype('uint8'))

    image = batch_sobel(x_train)
    edgemap = edgeMap.predict(image, batch_size = BATCH_SIZE)
    x_att_dataset = np.concatenate((x_train, edgemap), axis=3)
    y_att_dataset = (x_train - y_train)/2
    y_att_dataset = batch_rgb2gray_Contrast(y_att_dataset)

    try:
        #attentionMap = load_model("./models/attentionMap.h5")
        json_file = open('./models/attentionMap.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        attentionMap = tf.keras.models.model_from_json(loaded_model_json)
        attentionMap.load_weights("./models/attentionMap.h5")
        print("Load attentionMap from disk")
        attentionMap.compile(loss=edgeMSE, optimizer=tf.keras.optimizers.Adam(1e-4))
        #attentionMap.fit(x_att_dataset, y_att_dataset, batch_size=BATCH_SIZE, epochs=500, verbose=1)

        #attentionMap.save("./models/attentionMap.h5")
        model_json = attentionMap.to_json()
        with open("./models/attentionMap.json", "w") as json_file:
            json_file.write(model_json)
        attentionMap.save_weights("./models/attentionMap.h5")
        print("Saved attentionMap to disk")

    except:
        print("Train new attMap")
        attentionMap = attention_model()
        attentionMap.compile(loss=edgeMSE, optimizer=tf.keras.optimizers.Adam(1e-4))
        #attentionMap.fit(x_att_dataset, y_att_dataset, batch_size=BATCH_SIZE, epochs=1000, verbose=1)

        #attentionMap.save("./models/attentionMap.h5")
        model_json = attentionMap.to_json()
        with open("./models/attentionMap.json", "w") as json_file:
            json_file.write(model_json)
        attentionMap.save_weights("./models/attentionMap.h5")
        print("Saved new attentionMap to disk")

    image = batch_sobel(sample)
    edgemap = edgeMap.predict(image, batch_size = BATCH_SIZE)
    test_input = np.concatenate((sample, edgemap), axis=3)
    attention = attentionMap.predict(test_input, batch_size = BATCH_SIZE)
    image = np.squeeze(image)
    for i in range(image.shape[0]):
        io.imsave(image_save_path + "/sampleAttention"+ str(i) + ".jpeg", image[i].astype('uint8'))
    print("att saved")

    try:
        generator = load_model("./models/generator.h5")
        print("Load generator from disk")

        discriminator = load_model("./models/discriminator.h5")
        print("Load discriminator from disk")

        combined = load_model("./models/combined.h5", custom_objects={'my_gen_loss': my_gen_loss})
        print("Load combined from disk")

    except:
        print("Create new models")
        generator = generator_model()
        print("----------Generator----------")
        #generator.summary()

        discriminator = discriminator_model()
        print("----------Discriminator----------")
        #discriminator.summary()

        z = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL+2))
        #attention = attentionMap(z)
        #gen_in = layers.Concatenate(axis = 3)([z, attention])
        gen_in = z
        img = generator(gen_in)
        discriminator.trainable = False
        real = discriminator(img)
        combined = tf.keras.Model(z, [img, real])
        #combined = tf.keras.utils.multi_gpu_model(combined, gpus=GPU_NUM)
        print("----------CombinedModel----------")


        discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4))
        generator.compile(loss=MSE, optimizer=tf.keras.optimizers.Adam(1e-4))
        combined.compile(loss=[my_gen_loss, 'binary_crossentropy'], loss_weights=[1, 10], optimizer=tf.keras.optimizers.Adam(1e-4))

    output_true_batch = np.ones((BATCH_SIZE, 1))
    output_false_batch = np.zeros((BATCH_SIZE, 1))

    print("GANs Training...")
    for epoch in range(EPOCH):
        print("==>> Running epoch [{}/{}]...".format(epoch+1, EPOCH))
        start = time.time()

        d_loss = []
        g_loss = []
        for iter in range(int(x_train.shape[0] / BATCH_SIZE)):
            x_batch = x_train[iter*BATCH_SIZE : (iter+1)*BATCH_SIZE]
            y_batch = y_train[iter*BATCH_SIZE : (iter+1)*BATCH_SIZE]

            #attention_y_batch = (x_batch - y_batch)/2
            #attention_y_batch = batch_rgb2gray_Contrast(y_att_dataset)

            x_edge_batchset = batch_sobel(x_batch)
            batch_edge = edgeMap.predict_on_batch(x_edge_batchset)

            x_batch = np.concatenate((x_batch, batch_edge), axis=3)
            batch_attention = attentionMap.predict(x_batch, batch_size = BATCH_SIZE)
            x_batch = np.concatenate((x_batch, batch_attention), axis=3)

            gen_image = generator.predict_on_batch(x_batch)
            for item in range(5):
                d_loss_real = discriminator.train_on_batch(y_batch, output_true_batch)
                d_loss_fake = discriminator.train_on_batch(gen_image, output_false_batch)
            d_loss_batch = 0.5 * np.add(d_loss_real, d_loss_fake)
            g_loss_batch = combined.train_on_batch(x_batch, [y_batch, output_true_batch])

            print('Batch: %d, gen_loss: %f, dis_loss: %f' % (iter+1, np.average(g_loss_batch), d_loss_batch))
            d_loss.append(d_loss_batch)
            g_loss.append(np.average(g_loss_batch))

        generate_and_save_images(combined, edgeMap, attentionMap, epoch + 1, sample)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        if epoch != 0 and epoch % 2 == 0:
            save_models(attentionMap, generator, discriminator, combined)

    print("Training finished.")

def save_models(attentionMap, generator, discriminator, combined):
    #attentionMap.save("./models/attentionMap.h5")
    generator.save("./models/generator.h5")
    discriminator.save("./models/discriminator.h5")
    combined.save("./models/combined.h5")
    print("Models saved")

def batch_rgb2gray_Contrast(images):
    batch = []
    for item in images:
        im = rgb2gray(item)
        im = skimage.exposure.rescale_intensity(im)
        thresh = filters.threshold_otsu(im)
        pic = im > thresh
        im = skimage.exposure.rescale_intensity(pic)
        thresh = filters.threshold_otsu(im)
        pic = im > thresh

        im = rgb2gray(pic)* 127.5 + 127.5
        im = filters.sobel(im)
        im = skimage.exposure.rescale_intensity(im, in_range=(0, 1))
        batch.append(im * 255)

    batch = np.expand_dims(np.array(batch), axis=3)
    return batch

#process the Sobel operation
def batch_sobel(images):
    batch = []
    for item in images:
        im = rgb2gray(item)* 127.5 + 127.5
        im = filters.sobel(im)
        im = skimage.exposure.rescale_intensity(im, in_range=(0, 50))
        batch.append(im * 255)
    batch = np.expand_dims(np.array(batch), axis=3)
    return batch

def generate_and_save_images(model, edgeModel, attentionMap, epoch, test_input):
    test_edge_batchset = batch_sobel(test_input)
    edge_batch = edgeModel.predict_on_batch(test_edge_batchset)
    if epoch == 1:
        for i in range(edge_batch.shape[0]):
            io.imsave(image_save_path + "/edge"+ str(i) + ".jpeg", np.squeeze(edge_batch[i]).astype('uint8'))

    test_input = np.concatenate((test_input, edge_batch), axis=3)
    attention = attentionMap.predict(test_input, batch_size = BATCH_SIZE)
    test_input = np.concatenate((test_input, attention), axis=3)
    predictions = model.predict_on_batch(test_input)

    sample_result = predictions[0] * 127.5 + 127.5
    for i in range(sample_result.shape[0]):
        io.imsave(image_save_path + "/"+ str(i) +"-" + str(epoch) + ".jpeg", sample_result[i].astype('uint8'))


def MSSIMLoss(y_true, y_pred):
    return 1 - (tf.image.ssim_multiscale(y_true, y_pred, max_val=2.0))

def MSE(y_true, y_pred):
    return tf.losses.mean_squared_error(y_true*127.5+127.5, y_pred*127.5+127.5)

def edgeMSE(y_true, y_pred):
    return tf.losses.mean_squared_error(y_true, y_pred)

def perceptualLoss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(400, 400, 3))
    loss_model = tf.keras.Model(
        inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return tf.losses.mean_squared_error(loss_model(y_true*127.5+127.5), loss_model(y_pred*127.5+127.5))

def my_gen_loss(y_true, y_pred):
    return 10 * MSSIMLoss(y_true, y_pred) + 1 * MSE(y_true, y_pred) + 1 * perceptualLoss(y_true, y_pred)

def edge_model():
    inputs = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for iter in range(13):
        res = x
        x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, res])
        x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same')(x)
    edge_model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return edge_model

def attention_model():
    inputs = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL+1))

    mask_list = []

    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = layers.ReLU()(x)
    res = x

    for it in range(5):
        x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = layers.add([x, res])
        x = layers.ReLU()(x)
        res = x

    coni = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    i = layers.Activation('sigmoid')(coni)
    cong = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    g = layers.Activation('sigmoid')(cong)
    cono = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    o = layers.Activation('sigmoid')(cono)

    c_2 = layers.multiply([i, g])
    c = c_2
    c_act = layers.Activation('tanh')(c)
    h = layers.multiply([o, c_act])

    mask = layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same')(h)
    mask_list.append(mask)
    x = layers.Concatenate(axis = 3)([inputs, mask])


    for iter in range(3):

        x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = layers.ReLU()(x)
        res = x

        for it in range(5):
            x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
            x = layers.add([x, res])
            x = layers.ReLU()(x)
            res = x

        x = layers.Concatenate(axis = 3)([res, h])

        coni = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        i = layers.Activation('sigmoid')(coni)
        conf = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        f = layers.Activation('sigmoid')(conf)
        cong = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        g = layers.Activation('sigmoid')(cong)
        cono = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        o = layers.Activation('sigmoid')(cono)

        c_1 = layers.multiply([c, f])
        c_2 = layers.multiply([i, g])
        c = layers.add([c_1, c_2])
        c_act = layers.Activation('tanh')(c)
        h = layers.multiply([o, c_act])

        mask = layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same')(h)
        mask_list.append(mask)

    outputs = Lambda(lambda x: x)(mask)

    att_model = tf.keras.Model(inputs = inputs, outputs=outputs)
    #att_model = tf.keras.Model(inputs = inputs, outputs=[mask, mask_list])
    return att_model

def generator_model():
    inputs = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL+2))
    x = inputs

    '''
    #generator
    x = layers.Conv2D(64, (9, 9), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (9, 9), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    res1 = x
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    res2 = x
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    res3 = x
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same')(x)
    #x = layers.add([x, res3])
    x = layers.LeakyReLU()(x)
    x = layers.add([x, res3])
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same')(x)
    #x = layers.add([x, res2])
    x = layers.LeakyReLU()(x)
    x = layers.add([x, res2])
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same')(x)
    #x = layers.subtract([res1, x])
    x = layers.LeakyReLU()(x)
    x = layers.subtract([res1, x])
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (9, 9), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(3, (9, 9), strides=(1, 1), padding='same')(x)
    x = layers.Activation('tanh')(x)
    '''

    #generator
    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    res1 = x

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    res2 = x

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)


    #dilated convs
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=4)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=8)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=16)(x)
    x = layers.LeakyReLU()(x)


    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.add([x, res2])

    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.add([x, res1])

    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(3, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.Activation('tanh')(x)
    outputs = Lambda(lambda x: x)(x)

    gen_model = tf.keras.Model(inputs = inputs, outputs=outputs)
    return gen_model

def discriminator_model():
    inputs = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL))

    x = layers.Conv2D(8, (5, 5), strides=(1, 1), padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(16, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    mask = layers.Conv2D(1, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.multiply([x, mask])
    x = layers.Conv2D(64, (5, 5), strides=(4, 4), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(32, (5, 5), strides=(4, 4), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(1, (5, 5), strides=(4, 4), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    dis_model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return dis_model

def load_images():
    global TRAIN_IMAGES
    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, 'images/train/B')
    real_dir = os.path.join(current_dir, 'images/train/A')
    sample_dir = os.path.join(current_dir, 'images/test/B')
    sample2_dir = os.path.join(current_dir, 'images/test/A')

    if os.path.exists(image_save_path):
        shutil.rmtree(image_save_path)
    os.makedirs(image_save_path)

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
        imgArray = imgArray.astype('float32')
        imgArray = (imgArray - 127.5) / 127.5
        A_images.append(imgArray)
        index += 1

    print('%d pairs of image imported' % (index))
    TRAIN_IMAGES = index
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
        imgArray = imgArray.astype('float32')
        imgArray = (imgArray - 127.5) / 127.5
        B_images.append(imgArray)
        index += 1

    S_images = []
    for each in os.listdir(sample_dir):
        if each == ".DS_Store":
            continue
        img = Image.open(sample_dir + '/' + each)
        resizedImg = img.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
        imgArray = np.array(resizedImg)
        imgArray = imgArray.astype('float32')
        imgArray = (imgArray - 127.5) / 127.5
        S_images.append(imgArray)


    S_images2 = []
    for each in os.listdir(sample2_dir):
        if each == ".DS_Store":
            continue
        img = Image.open(sample2_dir + '/' + each)
        resizedImg = img.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
        imgArray = np.array(resizedImg)
        imgArray = imgArray.astype('float32')
        imgArray = (imgArray - 127.5) / 127.5
        S_images2.append(imgArray)

    return{
        'A': np.array(A_images),
        'B': np.array(B_images),
        'Sample': np.array(S_images),
        'Sample2': np.array(S_images2)
    }

if __name__ == '__main__':
    train()
