from itertools import groupby, islice, zip_longest, cycle, filterfalse
from matplotlib.widgets import Button
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Masking

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   


top = lambda img: 0
bottom = lambda img: int(img.shape[0])
hood = lambda img: bottom(img)*(1-theta['hood'])
left = lambda img: 0
right = lambda img: img.shape[1]
width = lambda img: right(img) - left(img)
height = lambda img: bottom(img) - top(img)
horizon = lambda img: int(img.shape[0]*theta['horizon'])
centerline = lambda img: int(img.shape[1]*0.5)
center = lambda img: [horizon(img), centerline(img)]
ground = lambda img: np.array([[[horizon(img), left(img)],
                                [horizon(img), right(img)],
                                [bottom(img), right(img)],
                                [bottom(img), left(img)]]])
sky = lambda img: np.array([[[top(img), left(img)],
                             [top(img), right(img)],
                             [bottom(img), right(img)],
                             [bottom(img), left(img)]]])
trapezoid = lambda img: np.array([[[horizon(img), centerline(img)-theta['trapezoid_top_factor']*width(img)/2],
                                   [horizon(img), centerline(img)+theta['trapezoid_top_factor']*width(img)/2],
                                   [hood(img), centerline(img)+theta['trapezoid_bottom_factor']*width(img)/2],
                                   [hood(img), centerline(img)-theta['trapezoid_bottom_factor']*width(img)/2]]]).astype(int)
trapezoid_pts = lambda img,m,b: ((int(x(hood(img),m,b)), int(hood(img))),
                                 (int(x(horizon(img),m,b)), int(horizon(img))))


def mask_image(img, vertices):
    return region_of_interest(img, vertices)


# Camera Calibration

def measure_distortion(calibration_files):
    files = calibration_files
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    stage1 = map(lambda x: (x,), cycle(files))
    stage2 = map(lambda x: x + (cv2.imread(x[0]),), stage1)
    stage3 = map(lambda x: x + (cv2.findChessboardCorners(cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY), (9,6)),), stage2)
    stage4 = map(lambda x: x + (cv2.drawChessboardCorners(np.copy(x[1]), (9,6), *(x[2][::-1])),), stage3)
    filenames,images,corners,annotated_images = zip(*filter(lambda x: x[2][0], islice(stage4, len(files))))
    _,imgpoints = zip(*corners)
    objpoints = [objp for i in range(len(imgpoints))]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, list(islice(stage2,1))[0][1].shape[:2:][::-1], None, None)
    return mtx, dist, annotated_images


# Distortion Correction

def get_undistorter(calibration_files):
    mtx,dist,annotated_images = measure_distortion(calibration_files)
    return lambda x: cv2.undistort(x, mtx, dist, None, mtx), annotated_images


# Perspective Transform

def measure_warp(corrected_image):
    def handler(e):
        if len(src)<1:
            plt.axvline(int(e.xdata), linewidth=2, color='r')
        if len(src)<4:
            src.append((int(e.xdata),int(e.ydata)))
        if len(src)<3:
            plt.axhline(int(e.ydata), linewidth=2, color='r')
        if len(src)==4:
            plt.axvline(int(e.xdata), linewidth=2, color='r')
            dst.extend([src[0], (src[0][0],src[1][1]), (src[3][0],src[2][1]), src[3]])
        else:
            pass
    plt.ion()
    fig = plt.figure()
    plt.imshow(corrected_image)
    global src
    global dst
    src = []
    dst = []
    cid1 = fig.canvas.mpl_connect('button_press_event', handler)
    cid2 = fig.canvas.mpl_connect('close_event', lambda e: e.canvas.stop_event_loop())
    fig.canvas.start_event_loop(timeout=-1)
    M = cv2.getPerspectiveTransform(np.asfarray(src, np.float32), np.asfarray(dst, np.float32))
    Minv = cv2.getPerspectiveTransform(np.asfarray(dst, np.float32), np.asfarray(src, np.float32))
    plt.ioff()
    return M, Minv


def get_unwarper(corrected_image):
    M, Minv = measure_warp(corrected_image)
    return lambda x: cv2.warpPerspective(x, M, x.shape[:2][::-1], flags=cv2.INTER_LINEAR)


def scale(img):
    scale_factor = np.max(img)/255
    return (img/scale_factor).astype(np.uint8)


def derivative(img):
    derivx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    derivy = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    gradmag = np.sqrt(derivx**2 + derivy**2)
    absgraddir = np.arctan2(np.absolute(derivy), np.absolute(derivx))
    return scale(derivx), scale(derivy), scale(gradmag), scale(absgraddir)


def threshold(img, thresh_min=0, thresh_max=255):
    binary_output = np.zeros_like(img)
    binary_output[(img >= thresh_min) & (img <= thresh_max)] = 1
    return binary_output


def pipeline2(img):
    img = np.copy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    h = hsv[:,:,0]
    l = hsv[:,:,1]
    s = hsv[:,:,2]
    derivx, derivy, gradmag, absgraddir = derivative(img)


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # img = mask_image(img, trapezoid(img)[:,:,::-1])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary


################################################################################

theta = {'horizon':0.61,
         'hood':0.07,
         'trapezoid_top_factor':0.10,
         'trapezoid_bottom_factor':0.90,
         'angle_cutoff':0.75,
         'kernel_size':5,
         'low_threshold':50,
         'high_threshold':150,
         'rho':2,
         'theta':1,
         'threshold':30,
         'min_line_length':3,
         'max_line_gap':1}

undistort, annotated_images = get_undistorter(glob.glob("camera_cal/*.jpg"))
plt.imshow(annotated_images[0])
plt.savefig("fig1.png", format="png", bbox_inches="tight")
plt.close()
distorted_image = plt.imread("test_images/straight_lines1.jpg")
corrected_image = undistort(distorted_image)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
f.tight_layout()
ax1.imshow(distorted_image)
ax1.set_title("Distorted Image")
ax2.imshow(corrected_image)
ax2.set_title("Corrected Image")
plt.savefig("fig2.png", format="png", bbox_inches="tight")
plt.close()

result = pipeline(corrected_image)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(corrected_image)
ax1.set_title('Original Image', fontsize=40)
ax2.imshow(result)
ax2.set_title('Pipeline Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

unwarp = get_unwarper(corrected_image)
flat_image = unwarp(corrected_image)
fig = plt.figure()
plt.imshow(flat_image)
plt.savefig("fig3.png", format="png")
plt.close()

