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

 
def label_segments(filename,savename='/testing_segment/',pwd=''):
    imgname = filename.split('/')[-1][:-4]
    image = io.imread(filename)
    gray_image = color.rgb2gray(image)
    thresh = filters.threshold_mean(gray_image)
    binary = gray_image > thresh
    bw_img = pwd+savename+imgname+'_binary.jpg'
    cv2.imwrite(bw_img,binary*255)
    # parse_image(filename)
    # parse_image(bw_img,grey=True)
    label_arr, num_seg = ndi.label(binary*255)     ##invert np.invert(binary))  
    segments = np.arange(1,num_seg+1)
    return binary,np.array(label_arr),segments,image

def plot_numbered_image(label_arr,savename='/testing_segment/',rotate=True,filepath=''):
    colors = np.array(list(chain(mcolors.TABLEAU_COLORS.values())))
    colors = np.repeat(colors,20000)                                         ### put in repeat for large sets
    if rotate: pixarray=np.rot90(label_arr,3)
    else: pixarray=label_arr
    imax,jmax = pixarray.shape
    fig,ax=plt.subplots(ncols=1, nrows=1, figsize=(20,int(20*jmax/imax)))
    plt.xticks(np.arange(0,imax))
    plt.yticks(np.arange(0,jmax))
    np.random.shuffle(colors)
    for i in range(0,imax):
        for j in range(0,jmax):
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

def parse_image(imagepath,grey=False):

    # binary_arr,label_arr, segments,image = label_segments(imagepath,'testing_segment/')
    # plot_numbered_image(label_arr,filepath=imagepath)
    print("processing",imagepath,"....")
    image = cv2.imread(imagepath)
    print(image.shape[0],image.shape[1])
    # if grey:
    #     image =cv2.merge([image,image])
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = filters.threshold_mean(gray)
    # binary = gray > thresh    
    blurred = cv2.blur(gray,(7,7))
    # circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 2, 50)
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
    cv2.destroyAllWindows()
    return

if __name__ == '__main__': 
    path = "/home/nina/logo-recognition-from-video/testing/"
    pwd =  "/home/nina/logo-recognition-from-video"
    images = os.popen("ls "+path).read().split('\n')[:-1]
    firstimage = images[5:6]
    for image in firstimage:
        imagepath = path+image
        label_segments(imagepath,pwd=pwd)
        # parse_image(imagepath)
    # testing = path+"single_circle.jpg"
    # print(testing)
    # parse_image(testing)