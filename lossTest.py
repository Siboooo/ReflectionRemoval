import tensorflow as tf
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import scipy
from PIL import Image

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=IMAGE_SHAPE)
    loss_model = Model(
        inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    #print("pLoss: "+str(K.mean(K.square(loss_model(y_true) - loss_model(y_pred))))+" ")
    return K.mean(K.square(loss_model(y_true*127.5+127.5) - loss_model(y_pred*127.5+127.5)))

if __name__ == '__main__':
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

    perceptual_loss(S_images2, S_images)
