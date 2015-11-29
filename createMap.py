import scipy.io as spio
import scipy.misc as spm
import matplotlib.pyplot as plt
import numpy as np

data_path = '/home/igor/data/trackData/'
pls_name = 'JET_58-425-73_D_CL_MX_3.9_1_26587_0'
data_fn = data_path + '/' + pls_name

import glob
fns = glob.glob(data_path + '/*.mat')

lats = []
lons = []

for i,fn in enumerate(fns):
    if i>2:
        break
    data = spio.loadmat(fn,struct_as_record=False, squeeze_me=True)
    lats.extend(data['gps'].lat)
    lons.extend(data['gps'].lon)

plt.plot(lons, lats, '.')

#basically, we should extend the axis a little (in each axes in order to avoid streching of the pic of map.

# axes
xlims = plt.xlim()
ylims = plt.ylim()

print xlims, ylims

height = 1080
width = 640
if height > 640:
    height = 640
if width > 640:
    width = 640

# Calculate zoom level for current axis limits

# function [x,y] = latLonToMeters(lat, lon )
# % Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"
# originShift = 2 * pi * 6378137 / 2.0; % 20037508.342789244
# x = lon * originShift / 180;
# y = log(tan((90 + lat) * pi / 360 )) / (pi / 180);
# y = y * originShift / 180;

def latLonToMeters(lat, lon):
     #inspired by the plot_google_map from matlab exchange
     #Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"
    originShift = 2 * np.pi * 6378137 / 2.0 # 20037508.342789244
    lon = np.array(lon)
    lat = np.array(lat)
    x = lon * originShift / 180
    y = np.log(np.tan((90 + lat) * np.pi / 360 )) / (np.pi / 180)
    y = y * originShift / 180
    return x, y

def curAxes2zoomLevel(xlims, ylims):
    xExtent, yExtent = latLonToMeters(ylims, xlims)
    minResX = (np.diff(xExtent) / width)[0]
    minResY = (np.diff(yExtent) / height)[0]
    minRes = max([minResX, minResY])
    tileSize = 256
    initialResolution = 2 * np.pi * 6378137 / tileSize; # 156543.03392804062 for tileSize 256 pixels
    zoomlevel = np.floor(np.log2(initialResolution/minRes))
    return zoomlevel

def get_ax_size(ax):
    fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height

autoAxis = True
if autoAxis:
    xExtent, yExtent = latLonToMeters(ylims, xlims)
    xExtent = (np.diff(xExtent))[0]
    yExtent = (np.diff(yExtent))[0]

    # get axes aspect ratio
    ax = plt.axes()
    pos1 = get_ax_size(ax)
    aspect_ratio = pos1[1] / pos1[0]

    if xExtent*aspect_ratio > yExtent:
        centerX = np.mean(xlims)
        centerY = np.mean(ylims)
        spanX = (xlims[1]-xlims[0])/2;
        spanY = (ylims[1]-ylims[0])/2;

        # enlarge the Y extent
        spanY = spanY*xExtent*aspect_ratio/yExtent # new span
        if spanY > 85:
            spanX = spanX * 85 / spanY
            spanY = spanY * 85 / spanY

        xlims = (centerX-spanX, centerX+spanX)
        ylims = (centerY-spanY, centerY+spanY)

    elif yExtent > xExtent*aspect_ratio:
        centerX = np.mean(xlims)
        centerY = np.mean(ylims)
        spanX = (xlims[1]-xlims[0])/2
        spanY = (ylims[1]-ylims[0])/2

        # enlarge the X extent
        spanY = spanX*yExtent/(xExtent*aspect_ratio) # new span
        if spanY > 180:
            spanY = spanY * 180 / spanX
            spanX = spanX * 180 / spanX

        xlims = (centerX-spanX, centerX+spanX)
        ylims = (centerY-spanY, centerY+spanY)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

xlims = plt.xlim()
ylims = plt.ylim()
zoomlevel = curAxes2zoomLevel(xlims, ylims)

# Enforce valid zoom levels
if zoomlevel < 0:
    zoomlevel = 0
if zoomlevel > 19:
    zoomlevel = 19

latC = np.mean(ylims)
lonC = np.mean(xlims)

scale = 1

# Construct query URL
str4pic = 'http://maps.googleapis.com/maps/api/staticmap'

str_center = '?center=%.8f,%.8f' %(latC, lonC)
str_zoom = '&zoom=%d' %(zoomlevel)
str_scale = '&scale=%d' %(scale)
str_size = '&size=%dx%d' %(width, height)

str4pic += str_center + str_zoom + str_scale + str_size

img_fn = 'ub'
import urllib
urllib.urlretrieve(str4pic, img_fn)
convertNeeded = 1

M = spm.imread(img_fn)
width = np.size(M, 1)
height = np.size(M, 0)

# Calculate a meshgrid of pixel coordinates in EPSG:900913
centerPixelY = np.round(height/2)
centerPixelX = np.round(width/2)
centerX, centerY = latLonToMeters(latC, lonC) # center coordinates in EPSG:900913
tileSize = 256
initialResolution = 2 * np.pi * 6378137 / tileSize; # 156543.03392804062 for tileSize 256 pixels

curResolution = initialResolution / 2**zoomlevel/scale; # meters/pixel (EPSG:900913)
xVec = centerX + (range(0,width)-centerPixelX) * curResolution # x vector
yVec = centerY + (range(height,0,-1)-centerPixelY) * curResolution # y vector
xMesh, yMesh = np.meshgrid(xVec,yVec) # construct meshgrid

def metersToLatLon(x, y):
    originShift = 2 * np.pi * 6378137 / 2.0 # 20037508.342789244
    lon = (x / originShift) * 180
    lat = (y / originShift) * 180
    lat = 180 / np.pi * (2 * np.arctan( np.exp( lat * np.pi / 180)) - np.pi / 2)
    return lon, lat

[lonMesh,latMesh] = metersToLatLon(xMesh,yMesh)



# % Calculate center coordinate in WGS1984
# lat = (curAxis(3)+curAxis(4))/2;
# lon = (curAxis(1)+curAxis(2))/2;
#
#
# latC = np.0.5*(np.min(lats)+np.max(lats))
# lonC = 0.5*(np.min(lons)+np.max(lons))
#
#
#
#
# plt.show()
#
#
# http://maps.googleapis.com/maps/api/staticmap?center=31.86557417,35.23952500&zoom=9&scale=1&size=680x1020