import time
from functools import partial
import cv2 as cv
import numpy as np

from cr_calibration import wholeFlow
from tm import globalTM, localTM, gaussianFilter, bilateralFilter, whiteBalance

# Demonstrate Overall Flow of HDR Imaging

# Declare results
radiance = None
radiance_wb = None
gtm = None
ltm = None
ltm_edge = None
# 1-1 Camera response calibration
radiance = wholeFlow('../TestImg/memorial', lambda_=50)
# 1-5 White Balance
if radiance is not None:
    ktbw = (457, 481), (400, 412)
    radiance_wb = whiteBalance(radiance, *ktbw)
# 1-2 Global tone mapping
if radiance_wb is not None:
    gtm = globalTM(radiance_wb)
# 1-3 Local tone mapping with Gaussian
if radiance_wb is not None:
    gauhw1 = partial(gaussianFilter, N=15, sigma_s=100)
    ltm = localTM(radiance_wb, gauhw1, scale=7)
# 1-4 Edge-Preserving filter
# Note that bilateral filter may be slow for large window size.
if radiance_wb is not None:
    bilhw1 = partial(bilateralFilter, N=15, sigma_s=100, sigma_r=0.8)
    ltm_edge = localTM(radiance_wb, bilhw1, scale=7)
print('Processed Done')
print('Press Any Key to Exit')


# Show image result
show_flag = False
if gtm is not None:
    cv.imshow('Global TM', gtm)
    show_flag = True
if ltm is not None:
    cv.imshow('Local TM', ltm)
    show_flag = True
if ltm_edge is not None:
    cv.imshow('Local TM-edge', ltm_edge)
    show_flag = True
if show_flag:
    cv.waitKey(0)
