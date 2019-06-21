# Single Image Reflection Remover

## Overview
Photographs were taken through a piece of transparent and reflective material, such as glass, often contain both the desired scene and the superimposed reflections. Reflection removal, as a computer vision task, aims at separating the mixture of the desired scene and the undesired reflections.

In this work, a novel attention based generative model is proposed to remove the reflection in an image with a multi-component loss function. The model learns to eliminate the reflection from a single-image input and transforms the reflection degraded image into a clean one. The networks consist of a generator network and a discriminator network which are both based on CNN architecture, and a Sobel edge prediction network (Edge-net) and an attentive-recurrent network (Attention-net) to provide supplementary information to the generator.

A new dataset $RSIR^{2}$ of 1,000 image pairs with and without reflections were collected and the dataset can be found [here](https://www.kaggle.com/c/digit-recognizer#description). 
Extensive experiments on a public benchmark dataset show the effectiveness of this approach, which performs favourably against state-of-the-art methods. 

## Details
Instead of using a Multi-image input, this project proposes a method which focuses on achieving the reflection removing goal based on single image input by utilising Generative Adversarial Networks (GANs). Similar to many GANs based image restoration techniques, such as denoising and super-resolution, the proposed GANs recovers a reflection-free image from a given image, which is supposed to be trained by image pairs with and without reflection, and learn to separate the desired scene layer and the reflection layer, i.e., eliminate the reflection from the single input image. The final goal for the GANs is to generate the background layer $T$ based on the input image $I$ and without any further input required, which means the proposed GANs only need a single image as input and it can generate some supplementary information by itself. This model contains a generator and a discriminator as fundamental components of GANs and a Sobel edge prediction network (Edge-net) and an attentive-recurrent network (Attention-net) to create additional information from the input image and provide them to the generator. During the training phase, two supplementary networks are trained first to provide qualified extra feature maps. And then the generator network and the discriminator network are trained in an adversarial manner to learn the elimination of the reflection based on images with reflections and the additional information from the other networks, i.e., the edge map obtained from the Edge-net and the attention map from the Attention-net.

Due to the vacancy of a large set of real-world reflection image pairs, in this work, the largest real reflection image dataset was collected and used as the training dataset. The dataset includes controlled scenes taken both indoor and outdoor with a great diversity of background scenes and reflections. Each pair of training data contains a reflection-free background image and an image with reflection. The dataset can be found [here](https://www.kaggle.com/c/digit-recognizer#description). 

## Performance
Image quality is continually improving follow the training. The Figure shows the output result in iteration 100, 300, 500, 800 and 1000.

![submission](https://github.com/Siboooo/imgForMD/blob/master/ReflectionRemoval/iterationSample.png?raw=true) 

Sample results in $SIR^2$ dataset]{Sample results in $SIR^2$ dataset. From left: Desired Scene, Input image, Sample Result

![submission](https://github.com/Siboooo/imgForMD/blob/master/ReflectionRemoval/SIRSample.png?raw=true) 

## Dependencies
* [NumPy](http://www.numpy.org)
* [SciPy](https://www.scipy.org)
* [Pandas](http://pandas.pydata.org)
