# Advanced Lane Detection

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
[calibrate]: ./output_images/calibrated_and_undistorted.png "Chessboard Calibration"
[distortion]: ./output_images/distortion_correction.png "distortion correction"
[combined]: ./output_images/combined.png "combined output"
[binary]: ./output_images/binary_warped_straight.png "binary output"
[histogram]: ./output_images/histogram.png "histogram output"
[window]: ./output_images/window.png "window search"
[polyfit]: ./output_images/polyfit.png "polyfit output"
[margin]: ./output_images/search_margin.png "search margin"
[pts]: ./output_images/left_right_pts_xfit.png "left and right points"
[final]: ./output_images/final.png "final output"


## Overview

Relevant Files and Folders:
* `advanced_lane_detection.ipynb`
* `lane_detection.py`
* `line.py`
* `video.py`
* `helpers.py`
* `project_out.mp4`
* `project_video.mp4`
* `output_images/`
* `camera_cal/`
* `test_images/`

The computer vision pipeline is defined in `lane_detection.py` in the `LaneDetection` class. It also imports `line.py` to store values in the `Line` class for each lane.  The pipeline is implemented with examples in the Jupyter Notebook: `advanced_lane_detection.ipynb`, which imports `helpers.py` for nice display functions. The final output is implemented again in `video.py` to generate the video.

Images referenced in this writeup are stored in `ouput_images/`, while images used for calibration and testing are stored in `camera_cal/` and `test_images/` respectively.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][calibrate]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][distortion]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of saturation channel (from HSL) and red channel (from RGB) isolation and binary thresholding, as well as x-oriented Sobel gradient thresholding to effectively identify the left and right lane lines.

This combined thresholding is done in the `combined_threshold()` of `lane_detection.py`, and an example implementation on a test image is shown below:


![alt text][combined]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transformation can be found in `lane_detection.py` in the function `warp_perspective()`.  It uses `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` to calculate the transformaton matrix and the warped images, respectively, using the `src` and `dst` values below:

```python
# Manually set some points for our perspective transformation
# four source coordinates
self.src = np.float32(
    [[690, 450],  # top right
     [1140, 720],  # bottom right
     [210, 720],  # bottom left
     [590, 450]])  # top left

# four desired coordinates
self.dst = np.float32(
    [[989, 0],  # top right
     [989, 720],  # bottom right
     [289, 720],  # bottom left
     [289, 0]])  # top left
```

The `src` points were chosen from carefully identifying pixels from observed images, while the `dst` points were selected by choosing a lane-width of 700, and centering it on the center of the image.  This allows 290px on either side of the lane.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The example below demonstrates both warping and combined thresholding.

![alt text][binary]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I did this in `lane_detection.py` in the function `polyfit_lanes()` to find the coefficients, then I calculated the x-coordinate for each y in `polyfit_lane_coords()`

![alt text][polyfit]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `lane_detection.py` in the function `measure_curvature()`.  

I started by calculating new left and right polyfitted coefficients using the pixel to meters conversion rates. Then i used the following radius of curvature function to calculate it for the left and right lanes:
```
Rcurve​= ((1+(2Ay+B)**2)**3/2)/∣2A∣

```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in my code in `lane_detection.py` in the function `find_lanes()`.  Here is an example of my result on a test image:

![alt text][final]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think there is a lot of room for improvement.  Even though the lane detection pipeline produces good results on the project video, it still lacking on the challenge videos.  

It seems to fail in extremely bright environments, or when there are additional long lines on the road that run parallel to the lane lines.  It also fails on sharp turns that cause one of the lane lines to temporarily disappear from view. 

I need to improve upon the combined thresholding techniques I'm currently using to better isolate yellow and white, vertical lines while also doing better at ignoring/seeing through changes in brightness.  Additionally, If the thresholding method was capable of adapting parameters to different environments, it could possibly produce much better results.