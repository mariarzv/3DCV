import os
import cv2
import numpy as np

def rnd(m):
    return np.round(m, decimals=5)

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
num_points = 10
points_3d = np.random.rand(num_points, 3)

# points project
points_2d_1, _ = cv2.projectPoints(points_3d, r, t1, k, None)
points_2d_2, _ = cv2.projectPoints(points_3d, r, t2, k, None)

# essential matrix
em, _ = cv2.findEssentialMat(points_2d_1, points_2d_2, k)

print("_____________________________________________problem 17")
print("essential matrix:\n", rnd(em))


########################################### problem 18 ###########################################

print("_____________________________________________problem 18")

t0 = np.array([[0.], [0.], [0.]])

rot1, rot2, tr = cv2.decomposeEssentialMat(em)
print("rot1:\n", rnd(rot1))
print("rot2:\n", rnd(rot2))
print("tr:\n", rnd(tr))

cam1r = (rot1, rot1, rot2, rot2)
cam1t = (tr, t0, tr, t0)
cam2r = (rot2, rot2, rot1, rot1)
cam2t = (t0, tr, t0, tr)

fourposes = zip(cam1r, cam1t, cam2r, cam2t,)

for cr1, ct1, cr2, ct2 in fourposes:
    # project points
    points_2d_img1, _ = cv2.projectPoints(points_3d, cr1, ct1, k, np.zeros((5,)))
    points_2d_img2, _ = cv2.projectPoints(points_3d, cr2, ct2, k, np.zeros((5,)))

    # filter out points that are not visible
    mask1 = cv2.inRange(points_2d_img1, np.array([0, 0]), np.array([image_width, image_height]))
    mask2 = cv2.inRange(points_2d_img2, np.array([0, 0]), np.array([image_width, image_height]))

    if np.sum(mask1) / 10 > 0.8 and np.sum(mask2) / 10 > 0.1:
        # correct cameras pose
        print("cr1: ", cr1, " ct1: ", ct1)

########################################### problem 19 ###########################################

print("_____________________________________________problem 19")

projdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img = cv2.imread(os.path.normpath(projdir + '/GOPR01170000.jpg'))

# principal point
height, width, _ = img.shape
cx, cy = width/2, height/2

# equivalent focal length in mm
eqf = 0.64 * 25.4

d35 = 43.27  # mm
sensor_size = 36.0  # mm (let's assume it is)

# real focal lenth px
fmm = eqf * d35 / sensor_size
fpx = fmm * sensor_size / width

# intrinsic matrix
kintr = np.array([[fpx, 0, cx],
                 [0, fpx, cy],
                 [0, 0, 1]])

print("intrinsic matrix:\n", rnd(kintr))
