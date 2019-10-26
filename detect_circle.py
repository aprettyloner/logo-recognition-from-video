import numpy as np
import argparse
import cv2
import os
from skimage import io
from skimage import filters, color
from scipy import ndimage as ndi
from itertools import chain
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt

 
def label_segments(filename,savename='/testing_segment/'):
    imgname = filename.split('/')[-1][:-4]
    image = io.imread(filename)
    gray_image = color.rgb2gray(image)
    thresh = filters.threshold_mean(gray_image)
    binary = gray_image > thresh
    io.imsave(savename+imgname+'_original.png',binary*255)

    label_arr, num_seg = ndi.label(np.invert(binary))
    segments = np.arange(1,num_seg+1)
    return binary,np.array(label_arr),segments,image

def plot_numbered_image(label_arr,savename='/testing_segment/',no_rotate=False,filepath=''):
    colors = np.array(list(chain(mcolors.TABLEAU_COLORS.values())))
    # np.repeat(colors,2)                                         ### put in repeat for large sets
    if no_rotate: pixarray=np.rot90(label_arr,3)
    else: pixarray=label_arr
    imax,jmax = pixarray.shape
    fig,ax=plt.subplots(ncols=1, nrows=1, figsize=(20,int(20*jmax/imax)))
    plt.xticks(np.arange(0,imax))
    plt.yticks(np.arange(0,jmax))
    np.random.shuffle(colors)
    for i in range(0,imax,10):
        for j in range(0,jmax,10):
            val = pixarray[i][j]
            if val != 0:
                ax.text(i,j,val,fontsize=20,color=colors[val])
    plt.xticks([])
    plt.yticks([])            
    imgname = filepath.split('/')[-1][:-4]
    pwd = os.popen("pwd").read().split('\n')[0]
    fig.savefig(pwd+savename+imgname+'_segmented.png')
    plt.close(fig)
    return

def parse_image(imagepath):

    # binary_arr,label_arr, segments,image = label_segments(imagepath,'testing_segment/')
    # plot_numbered_image(label_arr,filepath=imagepath)
    image = cv2.imread(imagepath)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 10000)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
        # show the output image
        cv2.imshow("output", np.hstack([image, output]))
        cv2.waitKey(0)


if __name__ == '__main__': 
    path = "/home/nina/video-image-recognition/testing/"
    images = os.popen("ls "+path).read().split('\n')[:-1]
    firstimage = images[:2]
    # for image in firstimage:
    #     imagepath = path+image
    #     print("processing ",imagepath,"....")
    #     parse_image(imagepath)
    testing = path+"circles.jpg"
    print(testing)
    parse_image(testing)