import numpy as np
import cv2

from lane_detection import LaneDetection

# Initiate The Lane Detection Pipeline
pipeline = LaneDetection()

name = 'project_'
# name = 'challenge_'
# name = 'harder_challenge_'

cap = cv2.VideoCapture(name + 'video.mp4')

# Check if camera opened successfully
if cap.isOpened() is False:
    print("Error opening video stream or file")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.mp4' file.
out = cv2.VideoWriter(name + 'out.mp4', cv2.VideoWriter_fourcc(*'H264'), 24, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        frame = pipeline.find_lanes(frame)

        left_curve_rad = pipeline.left.radius_of_curvature
        right_curve_rad = pipeline.right.radius_of_curvature

        rad_text = 'Radius of Curvature: ' + str(int(np.mean([left_curve_rad, right_curve_rad]))) + 'm'
        dist_test = 'Center Offset: ' + str(pipeline.center_offset) + 'm'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, rad_text, (50, 50), font, 1, (255, 255, 0), 2)
        cv2.putText(frame, dist_test, (50, 100), font, 1, (255, 255, 0), 2)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
