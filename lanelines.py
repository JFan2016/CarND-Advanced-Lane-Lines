from itertools import groupby, islice, zip_longest, cycle, filterfalse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

# Camera Calibration

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

files = glob.glob("camera_cal/*.jpg")
stage1 = map(lambda x: (x,), cycle(files))
stage2 = map(lambda x: x + (cv2.imread(x[0]),), stage1)
stage3 = map(lambda x: x + (cv2.findChessboardCorners(cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY), (9,6)),), stage2)
stage4 = map(lambda x: x + (cv2.drawChessboardCorners(np.copy(x[1]), (9,6), *(x[2][::-1])),), stage3)
_,_,corners,_ = zip(*filter(lambda x: x[2][0], islice(stage4, len(files))))
_,imgpoints = zip(*corners)
objpoints = [objp for i in range(len(imgpoints))]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, list(islice(stage2,1))[0][1].shape[:2:][::-1], None, None)

# Distortion Correction

# def cal_undistort(img, objpoints, imgpoints):
#     # Use cv2.calibrateCamera and cv2.undistort()
#     undist = np.copy(img)  # Delete this line
#     return undist

# undistorted = cal_undistort(img, objpoints, imgpoints)

# Color and Gradient Thresholding

def hls_select(img, thresh=(0, 255), channel=2):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,channel]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

# Perspective Transform
