import tensorflow as tf

IMAGE_SHAPE = (400, 540, 3)

def PSNR(yTar, yRes):
    return tf.image.psnr(yTar, yRes, max_val=1.0)

def SSIM(yTar, yRes):
    return tf.image.ssim(yTar, yRes, max_val=1.0)

def l1_loss(yTar, yRes):
    return tf.reduce_mean(tf.abs(yRes - yTar))

def wasserstein_loss(yTar, yRes):
    return tf.reduce_mean(tf.multiply(yTar, yRes))
