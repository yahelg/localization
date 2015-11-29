#!/usr/local/bin/python

import glob
import math as m
import numpy as np
from scipy import signal as sig
from scipy import interpolate as intrp

import scipy.io as spio
import matplotlib.pyplot as plt

all_curves_max_width = 10000
dist_sample_rate = 10.0


def nearest(array,value):
    return (np.abs(array-value)).argmin()


def load_data():
    mat_files = glob.glob('../drives/*.mat')[:50]
    all_curves = np.ones([len(mat_files), all_curves_max_width])*np.nan
    datas = []
    for i,mat_file in enumerate(mat_files):
        print mat_file
        data = spio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
        curve, dist = curvature_from_egomotion(data['ego_motion'].yaw, data['vehicleData'].speed)
        # print dist
        dt = 1.0/9.0
        old_dist = np.cumsum(data['vehicleData'].speed * dt)
        frames = data['frameIdx'][np.array([nearest(old_dist, d) for d in dist])]
        indexes = np.array([nearest(data['gps'].frameIdx, f) for f in frames])
        data['gps_lat'] = data['gps'].lat[indexes][:all_curves_max_width]
        data['gps_lon'] = data['gps'].lon[indexes][:all_curves_max_width]

        datas.append(data)
        if curve.shape[0] > all_curves_max_width:
            curve = curve[0:all_curves_max_width]
        all_curves[i,0:curve.shape[0]] = curve
    return datas, all_curves, mat_files


def interp_gps(datas):
    for data in datas:
        gps = data['gps']
        frameIdx = data['frameIdx']
        in_gps = (frameIdx >= gps.frameIdx[0]) & (frameIdx <= gps.frameIdx[-1])
        frameIdx = frameIdx[in_gps]
        speed = data['vehicleData'].speed[in_gps]
        dt = 1.0/9.0
        dist = np.cumsum(speed*dt)
        gps_frame_lat = intrp.interp1d(gps.frameIdx, gps.lat)(frameIdx)
        gps_frame_lon = intrp.interp1d(gps.frameIdx, gps.lon)(frameIdx)

        markers = np.linspace(dist[0],dist[-1],m.floor(dist[-1]/dist_sample_rate))

        gps_lat = intrp.interp1d(dist, gps_frame_lat)(markers)
        gps_lon = intrp.interp1d(dist, gps_frame_lon)(markers)
        data['gps_lat'] = gps_lat[:all_curves_max_width]
        data['gps_lon'] = gps_lon[:all_curves_max_width]


def running_ssd(a, b):
    if len(b) > len(a):
        a, b = b, a
    ssd = np.zeros(len(a) - len(b) + 1)
    for i in xrange(len(a) - len(b) + 1):
        ssd[i] = np.linalg.norm(a[i:i+len(b)] - b)
    return ssd


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

def plot_interesting(datas, all_curves):
    plt.figure(1)
    plt.figure(2)
    part = all_curves[22,900:1100]
    for i, curve in enumerate(all_curves):
        ssd = running_ssd(curve, part)
        ind = np.argmin(ssd)
        # if ssd[ind] < 0.01:
        #     print i
        plt.figure(1)
        plt.plot(running_ssd(curve, part))
        plt.figure(2)
        # plt.plot(datas[i]['gps'].lat, datas[i]['gps'].lon)
        plt.plot(datas[i]['gps_lat'][ind], datas[i]['gps_lon'][ind], '*')


def follow(datas, all_curves, num):
    plt.figure()
    for i, curve in enumerate(all_curves):
    # for i, curve in [(num, all_curves[num])]:
        lat = []
        lon = []
        for p in xrange(0, len(all_curves[num]), 100):
            part = all_curves[num][p:p+100]
            ssd = running_ssd(curve, part)
            ind = np.argmin(ssd)
            lat.append(datas[i]['gps_lat'][ind])
            lon.append(datas[i]['gps_lon'][ind])
        plt.plot(lat, lon, '*-')

    parts = all_curves[num]
# plt.imshow(all_curves)
# plt.show()
# datas, all_curves, mat_files = load_data()

# load_data()