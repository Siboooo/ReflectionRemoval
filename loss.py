import tensorflow as tf

IMAGE_SHAPE = (400, 400, 3)

def PSNR(yTar, yRes):
    return tf.image.psnr(yTar, yRes, max_val=1.0)

def SSIM(yTar, yRes):
    return tf.image.ssim(yTar, yRes, max_val=1.0)