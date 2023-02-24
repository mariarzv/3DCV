import os
import cv2
import numpy as np
import xml.etree.ElementTree as et

def rnd(m):
    return np.round(m, decimals=5)
########################################### problem 6 ############################################

# intrinsic parameters
fx = fy = 400
cx = 960
cy = 540

# extrinsic parameters
r6 = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0.],
              [np.sin(np.pi/4), np.cos(np.pi/4), 0.],
              [0., 0., 1.]])
t6 = np.array([0., 0., 10.]).reshape(-1, 1)

# projection matrix
k6 = np.array([[fx, 0., cx],
              [0., fy, cy],
              [0., 0., 1.]])
p6 = k6 @ np.hstack((r6, t6))

########################################### problem 10 ###########################################

print("_____________________________________________problem 10")
# chessboard cell points
cell_length = 0.2
objp = np.zeros((8*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:8].T.reshape(-1, 2) * cell_length

dist_coeffs = np.zeros((4, 1))  # no distortion
# projected
objpproj, j = cv2.projectPoints(objp, r6, t6, k6, dist_coeffs)

# PnP
x, rvec, tvec, e = cv2.solvePnPGeneric(objp, objpproj, k6, None, flags=cv2.SOLVEPNP_ITERATIVE, rvec=None, tvec=None, useExtrinsicGuess=False)
print("rvec:", rnd(rvec))
print("tvec:", rnd(tvec))

########################################### problem 12 ###########################################

print("_____________________________________________problem 12")

projdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img = cv2.imread(os.path.normpath(projdir + '/GOPR01170000.jpg'))
xmlfile = os.path.normpath(projdir + '/12camera.xml')

# find camera matrix
tree = et.parse(xmlfile)
root = tree.getroot()
data_node = root.find(".//camera_matrix/data")
dist_node = root.find(".//distortion_coefficients/data")
print(data_node)

# get camera matrix
intrinsic_matrix = np.fromstring(data_node.text, dtype=float, sep=' ')
intrinsic_matrix = intrinsic_matrix.reshape((3, 3))

# get distortion matrix
dist_matrix = np.fromstring(dist_node.text, dtype=float, sep=' ')

# image size
image_width = img.shape[1]
image_height = img.shape[0]

# undistort
undistorted_img = cv2.undistort(img, intrinsic_matrix, dist_matrix)

# and save the image (looks undistorted to me :) )
cv2.imwrite(os.path.normpath(projdir + '/undistimg.jpg'), undistorted_img)