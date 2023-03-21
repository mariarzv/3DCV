import numpy as np
import cv2

########################################### problem 5 ###########################################

print("_____________________________________________problem 5")
def closest_rank_r_matrix(A, r):
    U, s, Vt = np.linalg.svd(A)
    # calc rank of A as non-zero singular values in s
    if r <= np.sum(s > 0):
        # set singular values beyond rank r to zero
        s[r:] = 0
        S = np.zeros((U.shape[1], Vt.shape[0]))
        # construct S with truncated singular values, used this:
        # https://en.wikipedia.org/wiki/Low-rank_approximation
        # proof of EYM theorem for Frobenius norm
        S[:r, :r] = np.diag(s[:r])
        B = np.matmul(U, np.matmul(S, Vt))
        return B
    else:
        print("rank of B should be <= than rank of A! returning A:")
        return A


A = np.random.rand(5, 5)
r = 3
B = closest_rank_r_matrix(A, r)
print(B)
