import matplotlib.pyplot as plt
import cv2
import numpy as np
from lane_detection import LaneDetection


def side_by_side(images, labels=("l1", "l2"), cmap='viridis', cols=2):
    f, axes = plt.subplots(1, cols, figsize=(20,10))
    cmapv = cmap
    for idx in range(cols):

        if isinstance(cmap, str) is False:
            cmapv = cmap[idx]

        axes[idx].imshow(images[idx], cmap=cmapv)
        axes[idx].set_title(labels[idx], fontsize=30)


def visualize_perspective_transform(img):
    pipeline = LaneDetection()

    img = pipeline.distortion_correction(img)
    thresh = pipeline.combined_threshold(img)
    warped, M = pipeline.warp_perspective(thresh)

    # convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add src lines to image
    src_pts = np.array(pipeline.src, np.int32)
    src_pts = src_pts.reshape((-1, 1, 2))
    img = cv2.polylines(img, [src_pts], True, (255, 0, 0), 5)

    # add dst lines to image & convert to rgb so the lines are red
    thresh = np.array(np.dstack((thresh, thresh, thresh)) * 255, np.uint8)

    dst_pts = np.array(pipeline.dst, np.int32)
    dst_pts = dst_pts.reshape((-1, 1, 2))
    thresh = cv2.polylines(thresh, [dst_pts], True, (255, 0, 0), 5)

    warped = np.array(np.dstack((warped, warped, warped)) * 255, np.uint8)

    dst_pts = np.array(pipeline.dst, np.int32)
    dst_pts = dst_pts.reshape((-1, 1, 2))
    warped = cv2.polylines(warped, [dst_pts], True, (255, 0, 0), 5)

    # Visualize perspective transformation
    side_by_side(
        (img, thresh, warped),
        ('Original Image', 'Combined Threshold', 'Warped Perspective'),
        cols=3)
