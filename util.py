from itertools import groupby, islice, zip_longest, cycle, filterfalse
from matplotlib.widgets import Button
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


# Udacity helper functions

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask.astype('uint8'), vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# Define function y=f(x,m,b) and inverse function x=g(y,m,b).

y = lambda x,m,b: m*x+b
x = lambda y,m,b: (y-b)/m


# Define useful image features as functions.

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

# Define functions to get slopes and y-intercepts for an array of lines.

slope = lambda lines: (lines[:,0,3]-lines[:,0,1])/(lines[:,0,2]-lines[:,0,0])
intercept = lambda lines, m: lines[:,0,1]-m*lines[:,0,0]


# Define functions get indices into lines array, for left line and for right line.

lidx = lambda slopes: np.logical_and(np.isfinite(slopes),
                                     slopes<0,
                                     np.abs(slopes)>math.tan(theta['angle_cutoff']))
ridx = lambda slopes: np.logical_and(np.isfinite(slopes),
                                     slopes>0,
                                     np.abs(slopes)>math.tan(theta['angle_cutoff']))


# Define wrapper functions that adapt the Udacity helper functions.

def grayscale_image(img):
    return grayscale(img)


def blur_image(img):
    return gaussian_blur(img, theta['kernel_size'])


def edge_image(img):
    return canny(img, theta['low_threshold'], theta['high_threshold'])


def mask_image(img, vertices):
    return region_of_interest(img, vertices)
