from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import scipy.linalg

def rot2D(theta):
    return np.array((np.cos(theta), -np.sin(theta),np.sin(theta), np.cos(theta))).reshape((2,2))

def rot3D_YawPitchRoll(theta_x,theta_y,theta_z):
    raise NotImplementedError

def solve_psd(A,b,overwrite_b=False):
    return scipy.linalg.cho_solve(scipy.linalg.cho_factor(A),b,overwrite_b=overwrite_b)

potrs, potrf = scipy.linalg.lapack.get_lapack_funcs(('potrs','potrf'),arrays=False) # arrays=false means type=d
def solve_psd2(A,b,overwrite_b=False):
    return potrs(potrf(A,lower=False,overwrite_a=True,clean=False)[0],b,lower=False,overwrite_b=overwrite_b)[0]

def block_view(a,block_shape):
    shape = (a.shape[0]/block_shape[0],a.shape[1]/block_shape[1]) + block_shape
    strides = (a.strides[0]*block_shape[0],a.strides[1]*block_shape[1]) + a.strides
    return ast(a,shape=shape,strides=strides)
