#!/usr/local/bin/python

import glob
import math as m
import numpy as np
from scipy import signal as sig
from scipy import interpolate as intrp

import scipy.io as spio
import matplotlib.pyplot as plt

def curvature_from_egomotion(yaw,speed):

    dt = 1.0/9.0
    min_speed = 0.1
    medfilt_width = 25
    dist_sample_rate = 10.0

    speed = speed + min_speed
    dist_ = np.cumsum(speed*dt)
    curv_ = yaw/speed
    curv_mf = sig.medfilt(curv_,medfilt_width)
    curv_int_f = intrp.interp1d(dist_,curv_mf)
    dist = np.linspace(dist_[0],dist_[-1],m.floor(dist_[-1]/dist_sample_rate))
    curv = curv_int_f(dist)

    return curv, dist

all_curves_max_width = 2500
mat_files = glob.glob('../drives/*.mat')
all_curves = np.ones([len(mat_files),all_curves_max_width])*np.nan
for i,mat_file in enumerate(mat_files):

    print mat_file
    data = spio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    curve, dist = curvature_from_egomotion(data['ego_motion'].yaw, data['vehicleData'].speed)
    if curve.shape[0] > all_curves_max_width:
        curve = curve[0:all_curves_max_width]
    all_curves[i,0:curve.shape[0]] = curve
plt.imshow(all_curves)
plt.show()