import tensorflow as tf
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model

IMAGE_SHAPE = (400, 540, 3)

def PSNR(yTar, yRes):
    return tf.image.psnr(yTar, yRes, max_val=1.0)

def SSIM(yTar, yRes):
    return tf.image.ssim(yTar, yRes, max_val=1.0)

def l1_loss(yTar, yRes):
    return tf.reduce_mean(tf.abs(yRes - yTar))

def perceptual_loss(yTar, yRes):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=IMAGE_SHAPE)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return tf.reduce_mean(tf.square(loss_model(yTar) - loss_model(yRes)))

def wasserstein_loss(yTar, yRes):
    return tf.abs(tf.reduce_mean(tf.multiply(yTar, yRes)))

def perceptual_loss2(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(
        inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def wasserstein_loss2(y_true, y_pred):

    return K.mean(y_true - y_pred)
