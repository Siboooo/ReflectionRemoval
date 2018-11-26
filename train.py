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
