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
    return tf.reduce_mean(tf.abs(yTar, yRes))

def MSE(yTar, yRes):
    return tf.losses.mean_squared_error(yTar*127.5+127.5, yRes*127.5+127.5)

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=IMAGE_SHAPE)
    loss_model = Model(
        inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    #print("pLoss: "+str(K.mean(K.square(loss_model(y_true) - loss_model(y_pred))))+" ")
    return tf.losses.mean_squared_error(loss_model(y_true*127.5+127.5), loss_model(y_pred*127.5+127.5))
    #return K.mean(K.square(loss_model(y_true*127.5+127.5) - loss_model(y_pred*127.5+127.5)))


def wasserstein_loss(y_true, y_pred):
    #print("wLoss: "+K.mean(y_true * y_pred)+" ")
    return tf.abs(tf.reduce_mean(y_true - y_pred))

def discriminator_loss(real_output, generated_output):
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)
    total_loss = (real_loss + generated_loss)/2
    return total_loss

def dice_coefficient(y_true_cls, y_pred_cls):
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls )
    union = tf.reduce_sum(y_true_cls ) + tf.reduce_sum(y_pred_cls) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def singleDisLoss(y_true, y_pred):
    return tf.losses.sigmoid_cross_entropy(multi_class_labels=y_true, logits=y_pred)
