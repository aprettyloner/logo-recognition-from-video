# logo-recognition-from-video
Logo detection and recognition from advertisement videos using OpenCV and TensorFlow.



# Process

### 0) Obtain .mp4 advertisement videos
<img src='img/youtube_video.png'>Pull videos from youtube.</img>


### 1) Create .jpg frames from .mp4 video
<img src='img/frame1906_original.jpg'>Extract frames.</img>
<br>use frame_extractor.py

### 2) Create binary image
<img src='img/frame1906_binary.png'>Convert to binary to simplify segmentation.</img>
<br>use label_segments in detect_circle.py


### 3a) Blur image with CV2
<img src='img/cv2 blurring.png'>Use CV2 to blur image.</img>



### 3b) Apply Gaussian filter to iamge
<img src='img/filter_convolution.png'>Gaussian filter (5x5) on grayscale image.</img>


### 3) Detect circles
use parse_image in detect_cicle.py
