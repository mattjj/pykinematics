from __future__ import division
import numpy as np
import scipy.linalg

def rot2D(theta):
    return np.array((np.cos(theta), -np.sin(theta),np.sin(theta), np.cos(theta))).reshape((2,2))

def rot3D_YawPitchRoll(theta_x,theta_y,theta_z):
    raise NotImplementedError

def solve_psd(A,b,overwrite_b=False):
    return scipy.linalg.cho_solve(scipy.linalg.cho_factor(A),b,overwrite_b=overwrite_b)

def flatten1(lol):
    return reduce(list.__iadd__,lol,[])

