import math as m
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import interpolate as intrp

def curvature_from_egomotion(yaw,speed):

    dt = 1/9
    min_speed = 0.25
    dist = np.cumsum(speed*dt)
    curv = yaw/(speed+min_speed)
    curv_mf = sig.medfilt(curv,25)
    curv_int_f = intrp.interp1d(dist,curv_mf)
    dist_int = np.linspace(0,dist[-1],m.floor(dist[-1]/10))
    curv_int = curv_int_f(dist_int)
    return curv_int, dist_int

plt.plot(dist_int,curv_int,'.-')
plt.show()
