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


from sklearn import svm
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

colors = np.array(list(chain(mcolors.BASE_COLORS.values())))        ##mcolors.CSS4_COLORS

def parse_image(imagepath,grey=False):
    image = cv2.imread(imagepath)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.blur(gray,(7,7))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,20,blurred.shape[0]/64, param1=200, param2=10, minRadius=20, maxRadius=100)
    #cv2.GaussianBlur( gray, gray, (9, 9), 2, 2 )
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (255, 0, 0) , 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
        # show the output image
        cv2.imshow("output", np.hstack([image, output]))
        cv2.waitKey(0)
        return 1
    cv2.destroyAllWindows()
    return 0

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
    # firstimage = images[5:6]
    # for image in firstimage:
    #     imagepath = path+image
    #     process_image(imagepath,dirname=pwd)
    process_image("filter_convolution_square.png",dirname = '')