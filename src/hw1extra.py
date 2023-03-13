import numpy as np
import cv2

# approach - based on low rank approximation problem
# suppose matrix B's rank is less or equal to matrix A's rank
# SVD of A = USVᵀ
# use cv2.SVDecomp(A) to obtain those matrices
# rank of a matrix is equal to non-zero singular values in S  =>
# need to create B
# ||A||F of A

def closest_rank_r(A, r):
    if r > min(A.shape):
        print("rank value should be equal or less than A matrix dimension")
        print("changing rank to " + str(min(A.shape)))
        r = min(A.shape)
    U, S, Vt = cv2.SVDecomp(A)
    Sr = np.zeros_like(A)
    Sr[:r, :r] = np.diag(S[:r])
    Br = U.dot(Sr).dot(Vt)
    return Br


A = np.random.rand(5, 5)
print(closest_rank_r(A, 8))
