import cv2 
import subprocess
import os

# Function to extract frames 
def FrameCapture(path,filename): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path+filename) 
    outpath = "images/"+filename[:-4]
    # subprocess.run("mkdir images/"+"filename[:-4]")
    os.mkdir(outpath)

    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        cv2.imwrite(outpath+'/'+"frame%d.jpg" % count, image) 
  
        count += 1
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    list_of_ls = os.popen("ls videos/pepsi-01").read().split('\n')[:-1]
    print(list_of_ls)
    # FrameCapture("/home/nina/video-image-recognition/videos/pepsi-01/",list_of_ls[0]) 