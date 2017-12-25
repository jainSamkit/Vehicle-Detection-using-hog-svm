# Vehicle-Detection-using-hog-svm


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/bboxes.png
[image4]: ./output_images/test6.jpg
[image5]: ./output_images/output_bboxes.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/test1.jpg
[image8]: ./output_images/test2.jpg
[image9]: ./output_images/test3.jpg
[image10]: ./output_images/test4.jpg
[image11]: ./output_images/test5.jpg
[image12]: ./output_images/test6.jpg
[video1]: ./result_videos/project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code to extract hog features can be found in the lines 109-159 in the file `vehicle_detection_hog+svm.py`.

I started by reading the vehicle and non vehicle images from both KITTI GTI datasets.Initially,I tested my algorithm using only KITTI dataset but I had high false positives,.I tweaked various parameters ,to be discussed below,but the false positives were still high .Therefore,I introduced more datapoints from GTI folders as well.The code can be found in the lines 29-33 in the file `vehicle_detection_hog+svm.py`.

![alt text][image1]

I used different color features and parameters to decide what works best for detection.


#### 2. Explain how you settled on your final choice of HOG parameters.

There is a stark difference between car images and non-car images.The cars have a specific basic color, like ,Red or Yellow or black .And these can be captures with much confidence whereas the non car images have dull colors with less saturation.Therefore my first choice of color space was RGB which showed good result. I only used hog features to train my classifier as spatial and histogram features didnt add much value on top of HOG features as compared to the complexity added by them.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using only hog features collected from YCrCb color space and several other parameters in file `vehicle_detection_hog+svm.py` on lines 343-388.



`python code 
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 30  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL'# Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
n_frames=30
aspect_min,aspect_max = 0.7,3
min_bbox_area=50*50
min_close_area=80*80
close_y=500
close_x_thresh=450
`


![alt text][image2]


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I experimented with various windows of different sizes with different overlaps and at different positions. These could be seen in the lines 453-460 in the same python file mentioned above.

Then I searched for predictions in the window on the extracted hog features on them. I finally settled with the window size of (96,96) and overlap of 0.7 in X and Y directions.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I finally tested with the still test images from the test folders and got good results.One of the images is shown below and the rest is saved in output_images folder.To optimize the performance of the classifier,I restricted the windows to a particular size and overlap.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result_videos/project_output.mp4).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

I recorded the bounding box heat maps of last 30 frames and set the threshold to 33 heatp points, i.e the data points less than 33 points would automatically be set to zero! 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt_text][image5]

![alt text][image6]

### Here are six frames and their corresponding heatmaps:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hog features worked very nicely with SVM to detect and draw bounding boxes around the car at correct location.However,the pipeline is slow.With only 33 frames comes less accuracy and that too takes time to run on local machine.Apart,from this parameters such as x_start and the overlap area are hardcoded to work well on the video.However,the end to end learning approach using CNN's wouldnt require the hardcoding which can't be generalised.It was however great to realise the power of HOG's to seperate out specific images from the data.
