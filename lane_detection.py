import numpy as np
import cv2
import glob
import pickle
from line import Line


class LaneDetection:

    def __init__(self):
        self.corners_xy = (9, 6)
        self.mtx = None
        self.dist = None
        self.margin = 100
        self.ym_per_pix = 30 / 720   # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        self.center_offset = 0.0

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

        self.left = Line()
        self.right = Line()

    def calibrate_camera(self, imgs_dir, visualize=False):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.corners_xy[1] * self.corners_xy[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.corners_xy[0], 0:self.corners_xy[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(imgs_dir)

        # Expose variable for later
        img = None

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.corners_xy, None)

            # If found, add object points, image points
            if ret is True:
                objpoints.append(objp)
                imgpoints.append(corners)

                if visualize is True:
                    # Draw and display the corners_xy
                    cv2.drawChessboardCorners(img, self.corners_xy, corners, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)

        cv2.destroyAllWindows()

        # get the size from the 
        img_size = (img.shape[1], img.shape[0])
        # Do camera calibration given object points and image points
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # save to pickle
        dist_pickle = {
            "mtx": self.mtx,
            "dist": self.dist
        }
        pickle.dump(dist_pickle, open("camera_cal/dist_pickle.p", "wb"))

    def distortion_correction(self, img):
        if self.mtx is None or self.dist is None:
            try:
                # load mtx and dist from pickle file if it already exists
                dist_pickle = pickle.load(open("camera_cal/dist_pickle.p", "rb"))
                self.mtx = dist_pickle["mtx"]
                self.dist = dist_pickle["dist"]
            except (OSError, IOError) as e:
                print(e)
                print("distortion_correction() called before the camera has been calibrated with calibrate_camera()")

        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def warp_perspective(self, img, invert=False):
        # Define perspective transformation function

        img_size = (img.shape[1], img.shape[0])
        warped = None

        if invert is False:
            # Compute the perspective transform, M
            M = cv2.getPerspectiveTransform(self.src, self.dst)

            # Warp an image using the perspective transform, M:
            # Only necessary if not inverted
            warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        else:
            # Compute the inverse perspective transform (if you want to unwarp the image)
            # just swap the src and dst points in the function from above
            M = cv2.getPerspectiveTransform(self.dst, self.src)

        return warped, M

    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        if orient == 'x':
            sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # Apply threshold
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return grad_binary

    def mag_thresh(self, image, sobel_kernel=3, thresh=(0, 255)):
        # Calculate gradient magnitude
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Calculate the magnitude
        mag = np.sqrt(np.square(sobelx) + np.square(sobely))

        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255 * mag / np.max(mag))

        # Apply threshold
        mag_binary = np.zeros_like(scaled_sobel)
        mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return mag_binary

    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate gradient direction
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        grad_dir = np.arctan2(abs_sobely, abs_sobelx)

        # Apply threshold
        dir_binary = np.zeros_like(grad_dir)
        dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

        return dir_binary

    def r_threshold(self, image, thresh=(0, 255)):
        # image is expected to be in RGB format, so red is the third channel.
        r_channel = image[:, :, 0]

        # Threshold color channel
        r_binary = np.zeros_like(r_channel)
        r_binary[(r_channel >= thresh[0]) & (r_channel <= thresh[1])] = 1

        return r_binary

    def s_threshold(self, image, thresh=(0, 255)):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

        return s_binary

    def h_threshold(self, image, thresh=(0, 255)):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 0]

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

        return s_binary

    def gray_threshold(self, image, thresh=(0, 255)):
        # expects grayscale image
        gray_thresh = np.zeros_like(image)
        gray_thresh[(image >= 150) & (image <= 255)] = 1

        return gray_thresh

    def combined_threshold(self, image):
        # Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # set sobel_kernel size
        ksize = 5

        # Apply each of the thresholding functions
        # gray_binary = self.gray_threshold(gray, thresh=(100, 255))
        s_binary = self.s_threshold(image, thresh=(120, 255))
        r_binary = self.r_threshold(image, thresh=(90, 255))
        # h_binary = self.h_threshold(image, thresh=(20, 120))

        gradx = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(30, 100))
        # grady = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(50, 130))
        # mag_binary = self.mag_thresh(gray, sobel_kernel=ksize, thresh=(90, 140))
        # dir_binary = self.dir_threshold(gray, sobel_kernel=ksize, thresh=(0.0, 1.40))

        # combine them
        combined = np.zeros_like(s_binary)
        # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        # combined[(combined == 1) | (s_binary == 1)] = 1
        # combined[((s_binary == 1) & (dir_binary == 1)) | (gradx == 1)] = 1
        # combined[((s_binary == 1) & (r_binary == 1)) | ((gradx == 1) & (gray_binary == 1))] = 1
        combined[((s_binary == 1) & (r_binary == 1)) | (gradx == 1)] = 1
        # combined[((s_binary == 1) & (h_binary == 1) | (s_binary == 1) & (r_binary == 1))] = 1

        return combined

    def binary_warp_unwarp(self, img):
        # Correct for camera distortion
        img = self.distortion_correction(img)

        # Apply combined thresholding defined above
        thresh = self.combined_threshold(img)

        # Generate binary warped and the transformation matrix
        binary_warped, M = self.warp_perspective(thresh)

        # Generate the inverse for an unwarped perspective
        binary_unwarped, Minv = self.warp_perspective(thresh, invert=True)

        return (binary_warped, M), (binary_unwarped, Minv)

    def lane_histogram(self, binary_warped):
        # Take a histogram of the bottom half of the image
        return np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    def find_lane_pts(self, binary_warped, margin=100, nwindows=9, visualize=False):
        # Take a histogram of the bottom half of the image
        histogram = self.lane_histogram(binary_warped)

        if visualize is True:
            # Create an output image to draw on and  visualize the result
            out_img = np.array(np.dstack((binary_warped, binary_warped, binary_warped)) * 255, np.uint8)
        else:
            out_img = None

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if visualize is True:
                # Draw the windows on the visualization image
                out_img = cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                                        (0, 255, 0), 3)
                out_img = cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                                        (0, 255, 0), 3)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        pts = {
            'leftx': nonzerox[left_lane_inds],
            'lefty': nonzeroy[left_lane_inds],
            'rightx': nonzerox[right_lane_inds],
            'righty': nonzeroy[right_lane_inds]
        }

        return pts, out_img

    def polyfit_lanes(self, pts):
        # Fit a second order polynomial with x and y positions for each lane
        # self.left_fit = np.polyfit(pts["lefty"], pts["leftx"], 2)
        # self.right_fit = np.polyfit(pts["righty"], pts["rightx"], 2)
        self.left.set_polyfit(np.polyfit(pts["lefty"], pts["leftx"], 2))
        self.right.set_polyfit(np.polyfit(pts["righty"], pts["rightx"], 2))

    def polyfit_lane_coords(self, yrange):
        # For each y-coord, find x, for each lane
        ycoords = np.linspace(0, yrange - 1, yrange)
        # left_fitx = self.left.fit[0] * ycoords ** 2 + self.left.fit[1] * ycoords + self.left.fit[2]
        # right_fitx = self.right.fit[0] * ycoords ** 2 + self.right.fit[1] * ycoords + self.right.fit[2]
        self.left.set_xfitted(self.left.fit[0] * ycoords ** 2 + self.left.fit[1] * ycoords + self.left.fit[2])
        self.right.set_xfitted(self.right.fit[0] * ycoords ** 2 + self.right.fit[1] * ycoords + self.right.fit[2])

        return ycoords

    def measure_curvature(self, ycoords):
        # evaluate it against the bottom of the image, ie: nearest point on the road
        y_eval = np.max(ycoords)

        # Fit new polynomials to x,y in world space using the conversions (self.ym_per_pix & self.xm_per_pix)
        left_fit_cr = np.polyfit(ycoords * self.ym_per_pix, self.left.bestx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ycoords * self.ym_per_pix, self.right.bestx * self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.left.radius_of_curvature = ((1 + (
                2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        self.right.radius_of_curvature = ((1 + (
                2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

    def full_search(self, binary_warped):

        # Get the left/right lane points
        pts, _ = self.find_lane_pts(binary_warped)

        # fit second order polynomial to these points
        self.polyfit_lanes(pts)

        # generate coordinates
        ycoords = self.polyfit_lane_coords(binary_warped.shape[0])

        return ycoords

    def quick_search(self, binary_warped):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Find the arrays of indices within the margin
        left_lane_inds = ((nonzerox > (self.left.fit[0] * (nonzeroy ** 2) + self.left.fit[1] * nonzeroy +
                                       self.left.fit[2] - self.margin)) & (
                                  nonzerox < (self.left.fit[0] * (nonzeroy ** 2) + self.left.fit[1] *
                                              nonzeroy + self.left.fit[2] + self.margin)))

        right_lane_inds = ((nonzerox > (self.right.fit[0] * (nonzeroy ** 2) + self.right.fit[1] * nonzeroy +
                                        self.right.fit[2] - self.margin)) & (
                                   nonzerox < (self.right.fit[0] * (nonzeroy ** 2) +
                                               self.right.fit[1] * nonzeroy + self.right.fit[2] + self.margin)))

        # Again, extract left and right line pixel positions
        pts = {
            'leftx': nonzerox[left_lane_inds],
            'lefty': nonzeroy[left_lane_inds],
            'rightx': nonzerox[right_lane_inds],
            'righty': nonzeroy[right_lane_inds]
        }
        # Fit a second order polynomial to each
        self.polyfit_lanes(pts)
        # Generate x and y values for plotting
        ycoords = self.polyfit_lane_coords(binary_warped.shape[0])

        return ycoords

    def find_lanes(self, img):

        (binary_warped, M), (_, Minv) = self.binary_warp_unwarp(img)

        if self.right.fit is None and self.left.fit is None:
            ycoords = self.full_search(binary_warped)
        else:
            ycoords = self.quick_search(binary_warped)

        # Measure the lane curvature
        self.measure_curvature(ycoords)

        center = binary_warped.shape[1]/2
        # self.left.line_base_pos = np.absolute(center - self.left.xfit[719]) * self.xm_per_pix
        # self.right.line_base_pos = np.absolute(center - self.right.xfit[719]) * self.xm_per_pix
        self.center_offset = np.around((center - (self.left.xfit[719] + self.right.xfit[719])/2) * self.xm_per_pix, 4)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left.xfit, ycoords]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right.xfit, ycoords])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

        rad_text = 'Radius of Curvature: ' + \
                   str(int(np.mean([self.left.radius_of_curvature, self.right.radius_of_curvature]))) + 'm'
        dist_test = 'Center Offset: ' + str(self.center_offset) + 'm'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, rad_text, (50, 50), font, 1, (255, 255, 0), 2)
        cv2.putText(img, dist_test, (50, 100), font, 1, (255, 255, 0), 2)

        # Return the combined result with the original image
        return cv2.addWeighted(img, 1, newwarp, 0.3, 0)
