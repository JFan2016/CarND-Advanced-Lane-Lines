from itertools import groupby, islice, zip_longest, cycle, filterfalse
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Camera Calibration

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

files = glob.glob("camera_cal/*.jpg")
stage1 = map(lambda x: (x,), cycle(files))
stage2 = map(lambda x: x + (cv2.imread(x[0]),), stage1)
stage3 = map(lambda x: x + (cv2.findChessboardCorners(cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY), (9,6)),), stage2)
stage4 = map(lambda x: x + (cv2.drawChessboardCorners(np.copy(x[1]), (9,6), *(x[2][::-1])),), stage3)
filenames,images,corners,annotated_images = zip(*filter(lambda x: x[2][0], islice(stage4, len(files))))
_,imgpoints = zip(*corners)
objpoints = [objp for i in range(len(imgpoints))]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, list(islice(stage2,1))[0][1].shape[:2:][::-1], None, None)

plt.ioff()
plt.imshow(annotated_images[0])
plt.savefig("fig1.png", format="png", bbox_inches="tight")
plt.close()

# Distortion Correction

distorted_image = plt.imread("test_images/straight_lines1.jpg")
corrected_image = cv2.undistort(distorted_image, mtx, dist, None, mtx)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
f.tight_layout()
ax1.imshow(distorted_image)
ax1.set_title("Distorted Image")
ax2.imshow(corrected_image)
ax2.set_title("Corrected Image")
plt.savefig("fig2.png", format="png", bbox_inches="tight")
plt.close()

# Perspective Transform

def handler(e):
    if len(src)<4:
        src.append((int(e.xdata),int(e.ydata)))
    if len(src)<3:
        plt.axhline(int(e.ydata), linewidth=2, color='r')
    if len(src)>=4:
        dst.extend([src[0], (src[0][0],src[1][1]), (src[3][0],src[2][1]), src[3]])

plt.ion()
fig = plt.figure()
plt.imshow(corrected_image)
global src
global dst
src = []
dst = []
cid = fig.canvas.mpl_connect('button_press_event', handler)

M = cv2.getPerspectiveTransform(np.asfarray(src, np.float32), np.asfarray(dst, np.float32))
Minv = cv2.getPerspectiveTransform(np.asfarray(dst, np.float32), np.asfarray(src, np.float32))

warped_image = cv2.warpPerspective(corrected_image, M, corrected_image.shape[:2][::-1], flags=cv2.INTER_LINEAR)
plt.ioff()
fig = plt.figure()
plt.imshow(warped_image)
plt.savefig("fig3.png", format="png")
