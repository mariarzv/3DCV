import cv2
import numpy as np

########################################### problem 17 ###########################################

# intrinsic parameters
focal_length = 1000  # px
image_width = 640  # px
image_height = 480  # px
principal_point_x = image_width / 2
principal_point_y = image_height / 2
k = np.array([[focal_length, 0, principal_point_x],
              [0, focal_length, principal_point_y],
              [0, 0, 1]])


# rotation for both cameras
r = np.eye(3)

# translation for the first camera (no translation)
t1 = np.array([[0.], [0.], [0.]])

# translation for the second camera (moved 1 step along x-axis)
t2 = np.array([[1.], [0.], [0.]])

# points generate
num_points = 100
points_3d = np.random.rand(num_points, 3)
points_3d[:, 2] += 1.0

# points project
points_2d_1, _ = cv2.projectPoints(points_3d, r, t1, k, None)
points_2d_2, _ = cv2.projectPoints(points_3d, r, t2, k, None)

# essential matrix
E, _ = cv2.findEssentialMat(points_2d_1, points_2d_2, k)

print("problem 17 - essential matrix:\n", E)


########################################### problem 18 ###########################################