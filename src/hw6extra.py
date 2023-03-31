import cv2
import os
import numpy as np
from tempfile import mkdtemp
from util import getprojdir
from scipy.sparse import csr_matrix, block_diag


########################################### problem 20 ###########################################

print("_____________________________________________problem 20")

# C - Hilbert matrix of size 10x10 (the elements are Cij = 1 / (i + j -1)
C = np.array([[1 / (i + j - 1) for j in range(1, 11)] for i in range(1, 11)])
print("C", C.shape)

# Gb - blocks of: [[1,0,1],[0,1,0],[-1,0,1]]
Gb = np.array([[1, 0, 1], [0, 1, 0], [-1, 0, 1]])
print("Gb", Gb.shape)

# G - is block-diagonal matrix of size (3*10^6)x(3*10^6) that consists of Gb
# G = np.kron(np.eye(10**6, dtype=np.int8), Gb.astype(np.int8)) # memory error :(
diagonal_blocks = [csr_matrix(Gb) for i in range(10**6)]
# sparse matrix with diagonal blocks
G = block_diag(diagonal_blocks, format='csr')
print("G", G.shape)

# D - matrix of size 10x(3*10^6) consists of elements Dij = cos( i * j )
D = np.array([[np.cos(i * j) for i in range(1, 3 * 10**6 + 1)] for j in range(1, 11)])
print("D", D.shape)
print("D.T", D.T.shape)

vector_path = os.path.normpath(getprojdir() + '/rhs.csv')
r = np.loadtxt(vector_path, delimiter=",")
print("r", r.shape)

# assemble H
# H = np.block([[C, D.T.astype(C.dtype)], [D.astype(C.dtype), G]])
# print("H", H.shape)

# x = np.linalg.solve(H, r)
# print(x[:20])