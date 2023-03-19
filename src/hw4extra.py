import cv2
import os
import numpy as np
from util import getprojdir


# checkerboard size
board_size = (9, 7)

calibration_images = []
folder_path = os.path.normpath(getprojdir() + '/calibration')

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image = cv2.imread(os.path.join(folder_path, filename))
        calibration_images.append(image)

image_points = []
object_points = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
imagecount = 0

# find corner points
for i, image in enumerate(calibration_images):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    if ret and len(corners) >= 63:
        # append object points
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)
        object_points.append(objp)

        # corner pos refinement
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        image_points.append(corners)
        imagecount += 1
        print("image " + str(imagecount) + " processed")

    # can increase the cutoff value if you want more images processed
    # I set a small numbers because otherwise calculation time is huge!
    if i >= 60:
        break

# calibrating camera
ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, calibration_images[0].shape[:2], None, None)

# intrinsic camera parameters and distortion parameters
print("camera intrinsic matrix:\n", camera_matrix)
print("distortion coefficients:\n", distortion_coefficients)