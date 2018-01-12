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
[final]: ./output_images/final.png "final output"

![alt_text][final]

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

If you wish to read a description of how the algorithms work, as well as shortcoming and ideas for improvement,
please read [writeup.md](https://github.com/jyork03/advanced-lane-detection/blob/master/writeup.md).

## Running the Code

First clone the project down.

```bash
git clone git@github.com:jyork03/advanced-lane-detection.git
```

Running it with Docker:

```bash
docker pull udacity/carnd-term1-starter-kit
cd ./advanced-lane-detection
docker run -it --rm -v `pwd`:/src -p 8888:8888 udacity/carnd-term1-starter-kit
```

If you don't use docker, you'll need to install the dependencies yourself.

The dependencies include:
* python==3.5.2
* numpy
* matplotlib
* jupyter
* opencv3
* ffmpeg