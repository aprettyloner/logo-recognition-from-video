from skimage import io
from skimage import filters, color
from scipy import ndimage as ndi
from PIL import Image
import cv2

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,6)

from scipy.misc import imresize
from itertools import chain
from math import ceil 
import subprocess
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timeit

import tensorflow as tf
from tensorflow.keras import layers
keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE ## tf.data transformation parameters
import matplotlib
matplotlib.style.use('ggplot')


from sklearn import svm
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

colors = np.array(list(chain(mcolors.BASE_COLORS.values())))        ##mcolors.CSS4_COLORS


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [576, 576])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def apply_convolution_to_image(image,
                               convolutional_filter, 
                               image_shape=(576, 576),
                               filter_shape = (5, 5)):
    """Apply a convolutional filter to an image.  The inputs here should be
    numpy arrays, this function will take care of converting them to tensors
    and back.
    """
    # The image and filter tensor must be 4-tensors to use conv2d.  This
    # will eventually make sense, as we build up the complexity of our
    # filters.
    image_tensor = np.array(image).reshape(1, image_shape[0], image_shape[1], 1)
    filter_tensor = convolutional_filter.reshape(filter_shape[0], filter_shape[1], 1, 1)
    convolved_tensor = tf.nn.conv2d(input=image_tensor, 
                                    filters=filter_tensor, 
                                    strides=[1, 1, 1, 1], 
                                    padding='SAME')
    image_convloved = convolved_tensor[0, :, :, 0]
    return image_convloved



def parse_image(imagepath,grey=False):
    raw_image = tf.io.read_file(imagepath)
    # image = tf.image.decode_jpeg(raw_image, channels=3)   
    
    fig,axes = plt.subplots(1,3)
    
    image = preprocess_image(raw_image)
    axes[0].imshow(image)
    axes[0].set_title('Original')

    # output = image.copy()
    gray = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    axes[1].imshow(gray)
    axes[1].set_title('Grayscale')

    blurred = cv2.blur(np.array(gray),(3,3))
    axes[2].imshow(blurred)
    axes[2].set_title('Blurred')


    gaussian_filter = (1/255.0) * np.array([
            [1, 4,  6,  4,  1], 
            [4, 16, 24, 16, 4], 
            [6, 24, 36, 24, 6], 
            [4, 16, 24, 16, 4], 
            [1, 4,  6,  4,  1]
        ]).astype(np.float32)

    for ax in axes.flatten():
        ax.grid(False)
    plt.show()

    first_digit_blurred = apply_convolution_to_image(tf.image.rgb_to_grayscale(image), gaussian_filter)

    fig, axs = plt.subplots(1, 3, figsize=(10, 6))

    plot_convolution(image, gaussian_filter, first_digit_blurred, axs)
    for ax in axs.flatten():
        ax.grid(False)
    plt.show()
    return 0


def plot_convolution(before, convolutional_filter, after, axs):
    """Plot a 1 by 3 grid of images:
        - A before image.
        - A filter to apply.
        - The result of convolving the filter with the image.
    """
    axs[0].imshow(before, cmap=plt.cm.gray_r, interpolation="nearest")
    axs[0].set_title("Before Convolution")
    axs[1].imshow(convolutional_filter, cmap=plt.cm.gray_r, interpolation="nearest")
    axs[1].set_title("Filter")
    axs[2].imshow(after, cmap=plt.cm.gray_r, interpolation="nearest")
    axs[2].set_title("After Convolution")

def count_seg(filename):
    image = io.imread(filename)
    gray_image = color.rgb2gray(np.invert(image))
    thresh = filters.threshold_mean(gray_image)
    binary = gray_image > thresh
    label_arr, num_seg = ndi.label(np.invert(binary))
    return num_seg

def label_segments(filename,savename='/testing_segment/',pwd=''):
    imgname = filename.split('/')[-1][:-4]
    image = io.imread(filename)
    gray_image = color.rgb2gray(image)
    thresh = filters.threshold_mean(gray_image)
    binary = gray_image > thresh
    bw_img = pwd+savename+imgname+'_binary.jpg'
    cv2.imwrite(bw_img,binary*255)
    label_arr, num_seg = ndi.label(binary*255) 
    segments = np.arange(1,num_seg+1)
    return binary,np.array(label_arr),segments,image


def crop_image(segment,label_arr,binary_arr,ax=None,plot=False,model=None,direc='export/',svc=False):
    nrows = 28
    ncolumns = 28
    found = label_arr == segment    
    # plt.imshow(found)
    seg = direc+ 'testseg'+str(segment)+'.jpg'
    io.imsave(seg,found*1)
    status = parse_image(seg)

    


def process_image(filename,dirname,fitted_clf=None,plot=False,svc=False):
    binary_arr,label_arr, segments,orig = label_segments(filename,dirname)
    for segment in segments:
        crop_image(segment,label_arr,binary_arr)
    # plot_numbered_image(label_arr,dirname)

if __name__ == '__main__': 
    path = "/home/nina/logo-recognition-from-video/testing/"
    pwd =  "/home/nina/logo-recognition-from-video"
    images = os.popen("ls "+path).read().split('\n')[:-1]
    firstimage = images[5:6]
    for image in firstimage:
        imagepath = path+image
        # process_image(imagepath,dirname=pwd)
        parse_image(imagepath)