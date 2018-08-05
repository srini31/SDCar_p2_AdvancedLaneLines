## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images_videos/test_1.jpg  "Distortion test image"
[image2]: ./output_images_videos/undistorted_test_1.jpg "Undistorted image"
[image3]: ./output_images_videos/binary_1.jpg "Binary image 1"
[image4]: ./output_images_videos/binary_2.jpg "Binary image 2"
[image5]: ./output_images_videos/lane_perspective_1_good.jpg "Perspective of binary lane"
[image6]: ./output_images_videos/lane_perspective_2_noise.jpg "Perspective of binary lane - noise"
[image7]: ./output_images_videos/filled_lane_perspective_1_good.jpg "Filled Perspective Lane"
[image8]: ./output_images_videos/filled_lane_perspective_2_error.jpg "Filled Perspective Lane - some error"
[image9]: ./output_images_videos/filled_lane_perspective_3_bad.jpg "Filled Perspective Lane - bad"
[image10]: ./output_images_videos/final_image_1_good.jpg "Final filled lane - good"
[image11]: ./output_images_videos/final_image_2_some_error.jpg "Final filled lane - some error"

[video1]: ./output_images_videos/submit_output_video.mp4 "Project Output video"
[video2]: ./output_images_videos/submit_challenge_video.mp4 "Challenge video"
[video3]: ./output_images_videos/submit_hard_challenge_video.mp4 "Hard Challenge video"



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first cell `# Code Part 1` of the IPython notebook located in "./AdvancedLaneFinding.ipynb" 


I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![Distorted image][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
I used the matrix(mtx) and distortion coefficients (dst) from the above step and applied the `cv2.undistort()` function.
![Undistorted image][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The thresholding code is in the cell marked with `# Code Part 3` of the notebook.  Here's an example of my output for this step. 

I have code for finding magnitude and direction gradient for gray images, R, G, B values and also H, S, V values.
Gray and R channels are good for right dotted line but not good if road color is white on the left side (on bridge)
R, S and H are good for yellow lane (on the left or right of road). I finally used a binary operation which is illustrated below.
combined_binary[( ((rgb_r_binary == 1) | (rgb_g_binary == 1)) | (((hls_h_binary == 1) | (hls_s_binary == 1)) & (dir_binary == 1)))] = 1

First Red(R) and Green(G) channels, Hue(H) and Saturation(S) channels are used in a union (OR) operation. Then the result of these two is also put in a OR operation and the final result is intersected (AND) with the direction binary operation. I expected the direction binary could be especially helpful in the harder challenge video and also when the lane lines are dotted. This choice was made based on the tests I ran for a series of images and felt that this is a good approximation. There is room to improve this and I mention this in the discussion section. Here are the results.

![Binary image 1][image3]
![Binary image 2][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the code cell `# Code Part 5` of the notebook. 

The `getPerspectiveTransform` takes input of source (`src`) and destination (`dst`) points and returns a matrix. An inverse tranform matrix is also generated by reversing the order of parameters `src` and `dst` so that the image can be re-transformed to the real perspective of the road. The `warpPerspective` function takes the image and transformation matrix as input and returns the tranformed image. Here is the code I used

```python
wid = img.shape[1]     # 1280
hgt = img.shape[0]     # 720
img_size = (wid, hgt)  # img_size = (gray.shape[1], gray.shape[0])  # Grab the image shape

wid_off = 0.25*wid   # width offset
hgt_off = 0.64*hgt     # height offset
bottom_trim = .96 #.94 -- can be more as dashboard is curved
#### source points 
#4;3  -- order of points
#1;2
src = np.float32([ [wid_off*0.95, hgt*bottom_trim], [3*wid_off*1.1, hgt*bottom_trim], 
                   [0.55*wid, hgt_off ], [0.45*wid, hgt_off] ])
#### destination points
dst = np.float32([ [wid_off*1.1,hgt], [3*wid_off*0.9, hgt], [3*wid_off*0.9, 0], [wid_off*1.1, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 304, 691      | 352, 720      | 
| 1056, 691     | 864, 720      |
| 704, 460      | 864, 720      |
| 576, 460      | 352, 0        |
|:-------------:|:-------------:| 

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Perspective of binary lane][image5]
![Perspective of binary lane - noise][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


I used code from the Advanced Computer Vision/Another sliding window approach quiz for this part. The code is listed in a cell titled `Sliding window using convolution` (`# Code Part 4`). 

Instead of using the histogram function, this code tries to find the local peaks at the bottom left and right of the image using convolution. I used a window width, height and margin variables to create areas of search for the lane pixels and iterate through the image. I calculated the max and min index of the left lane pixels and used the conv_signal method and then further calculated the center of the window where the white pixels of the lane are found. This was repeated for the right lane pixels as well.

I did not understand this 100% and need to look at this in detail. The polynomial fitting was easier to understand but more cumbersome.
I mainly need to check why the conv_signal method returns the center to the left of the window if it does not find a lane pixel. I guess averaging the centers is a good idea to prevent this but I was not able to implement that code. 

Below are some images of the detected lanes in the perspective image. The windows are colored with dark green and the area between them is 
filled with a light green color. For this I filled all the pixels between the detected windows on the lane with a lighter green color 

![Filled Perspective Lane][image7]
![Filled Perspective Lane - some error][image8]
![Filled Perspective Lane - bad][image9]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for my perspective transform is in the code cell `# Code Part 5` of the notebook. I fitted the polynomial based on the detected left and right lane pixels and colored the lanes. The radius of curvature was calculated as per the instructions in the quiz in the Advanced lane finding class. I used the `Self-Driving Car Project Q&A` video as reference in this section. The radius of curvature and distance from center are displayed on the image.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The lanes were calculated from the polynomial and this image was warped using the inverse transform matrix. The original image was overlaid with the detected lanes and also the the image from part (4) above was also transformed and added to it using the `cv2.addWeighted` method.

![Final filled lane - good][image10]
![Final filled lane - some error][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here are the links to my output videos for the project. All my output files are located in the `output_images_videos` folder. The code for the project output, challenge and harder challenge videos are in the cells titled `# Code Part 7`, `# Code Part 9`, `# Code Part 11`

The code performs well on the project video but the lanes go off the road sometimes in the challenge video. The harder challenge video is even difficult and many updates have to be made for that.


![Project Output video][video1]
![Challenge video][video2]
![Hard Challenge video][video3]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Since the first submission, I have made some updates to overcome some of the problems 

Issues documented in the first submission:
1) I did not implement averaging of centers of pixels during convolution. Because of this when there are dotted lanes or missing lanes, the pixels are shifting to the left. By solving this I can make the lanes follow the road more smoothly.

2) Sometimes there is a reflection of the lane line on the side of the car that is overtaking the self driving car. This reflected lane is detected and causes wobble of the lane lines.

3) I am not sure how to save the previous lane detection when the video clip calls the `advanced_lane_find` in a new frame. May be this has to be done using a pickle file to load the previous position. I am also not sure about the ratios for converting the pixel size to meters.

Solutions to the issues in the first submission:
1) updated , the left and right start points of the lanes are averaged and if the lane starting points drift more than 25 points, the average value is used
2) The window centroids are averaged for four frames (36 points) and the average centroids are returned from the `sliding_window_conv` class. This reduces the lane drift and is also helpful when some pixels are not detected in some frames. 
3) The leftx and rightx arrays are also averages for eight frames and hence the polynomial is also averaged.
2) and 3) almost eliminated the issue from the earlier submission where the lane was moving inward near the bridge and outward when the black car was overtaking the self driving car as highlighted in the review.

#----------------------------------------------------------
For the challenge video, there are some lines in the center of the road which are detected and followed by the algorithm. I need to correct this by looking at the binary transformation. The current binary image combination is working but I need to adjust the source points array for this video as the lane positions are different compared to the project video. 

For the harder challenge video, I need to reduce the region of interest to better follow the steep curves. Though the lanes seem to be clearly marked, the lines seem to jump on to the railing on the side of the road. May be I need to use a margin to detect the lanes well instead of going out of the road.

Most of these issues make me wonder about the real challenges of the self driving car. Things can be hard when there are two dotted lanes and the perspective may not be able to figure the direction. The window height may have to reduced, centers averaged, and make sure that the two detected polynomials are parallel to each other, more smoothing has to be applied in the convolution program. From the harder challenge video, bright sunlight, shiny surfaces are more prominent than the lanes and the steep curves cause the perspective image to veer on to the side of the image whereas the sliding windows are marching upwards in the convolution program. For these cases, the margin of search window has to be wider.





References used:
- Code from the quizzes in the class
- Self-Driving Car Project Q&A | Advanced Lane Finding (https://www.youtube.com/watch?v=vWY8YUayf9Q&feature=youtu.be)
