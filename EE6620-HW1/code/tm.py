# Functions for tone mapping
# Put all your related functions for section 1-2~1-5 in this file.
import numpy as np
import cv2 as cv
import math
import sys
from functools import partial
gamma = 2.2


def globalTM(src, scale=1.0):
    """Global tone mapping (section 1-2)

    Args:
        src (ndarray, float32): source radiance image
        scale (float, optional): scaling factor (Defaults to 1.0)
    """
    result = np.zeros_like(src, dtype=np.uint8)
    
    # find max_rad in each channel
    max_rad = [-1,-1,-1]
    for channel in range(0, result.shape[2], 1):
        for i in range(0, result.shape[0], 1):
            for j in range(0, result.shape[1], 1):
                if max_rad[channel] < src[i][j][channel]:
                    max_rad[channel] = src[i][j][channel]

    for channel in range(0, result.shape[2], 1):
        for i in range(0, result.shape[0], 1):
            for j in range(0, result.shape[1], 1):
                try:
                    adjustment = scale*(math.log(src[i][j][channel],2)-math.log(max_rad[channel],2))+math.log(max_rad[channel],2)
                except:
                    adjustment = scale*(-10000000-math.log(max_rad[channel],2))+math.log(max_rad[channel],2)
                compressed_result = math.pow(2, adjustment)
                gamma_correction = math.pow(compressed_result, 1/gamma)
                gamma_correction = gamma_correction * 255
                if gamma_correction > 255:
                    gamma_correction = 255
                elif gamma_correction < 0:
                    gamma_correction = 0
                result[i][j][channel] = gamma_correction
    
    return result


def localTM(src, imgFilter, scale=3):
    """Local tone mapping (section 1-3)

    Args:
        src (ndarray, float32): source radiance image
        imgFilter (function): filter function with preset parameters
        scale (float, optional): scaling factor (Defaults to 3)
    """
    LB = imgFilter(src)
    result = np.zeros_like(src, dtype=np.uint8)
    return result


def gaussianFilter(src, N=35, sigma_s=100):
    """Gaussian filter (section 1-3)

    Args:
        src (ndarray): source image
        N (int, optional): window size of the filter (Defaults to 35)
            filter indices span [-N/2, N/2]
        sigma_s (float, optional): standard deviation of Gaussian filter (Defaults to 100)
    """
    # Window size should be odd
    assert N % 2
    dtype = np.float32
    try:
        I = (src[:,:,0] + src[:,:,1] + src[:,:,2])/3
        result = np.zeros_like(I, dtype=dtype)
        print(I.shape)
        print(result.shape)
        L = np.zeros(I.shape)
        for i in range(0, I.shape[0], 1):
            for j in range(0, I.shape[1], 1):
                try:
                    L[i][j] = math.log(I[i][j], 2)
                except:
                    L[i][j] = sys.float_info.min
    except:
        result = np.zeros_like(src, dtype=dtype)
        I = src
        L = I
    # Pad the Image, Assume Square filter
    pdsize = int(N/2)
    padded = np.pad(L, ((pdsize, pdsize), (pdsize, pdsize)), 'symmetric')
    for i in range(pdsize, padded.shape[0] - pdsize, 1):
        for j in range(pdsize, padded.shape[1] - pdsize, 1):
            num1 = 0
            num2 = 0
            for window_k in range(i - pdsize, i + pdsize + 1, 1):
                for window_l in range(j - pdsize, j + pdsize + 1, 1):
                    w = math.exp(-((i-window_k)**2+(j-window_l)**2)/(2*sigma_s**2))
                    num2 += w
                    num1 = num1 + padded[window_k][window_l] * w
            result[i-pdsize][j-pdsize] = num1/num2
    return result
    

def bilateralFilter(src, N=35, sigma_s=100, sigma_r=0.8):
    """Bilateral filter (section 1-4)

    Args:
        src (ndarray): source image
        N (int, optional): window size of the filter (Defaults to 35)
            filter indices span [-N/2, N/2]
        sigma_s (float, optional): spatial standard deviation of bilateral filter (Defaults to 100)
        sigma_r (float, optional): range standard deviation of bilateral filter (Defaults to 0.8)
    """
    # Window size should be odd
    assert N % 2
    dtype = np.float32
    result = np.zeros_like(src, dtype=dtype)
    return result


def whiteBalance(src, y_range, x_range):
    """White balance based on Known to be White(KTBW) region

    Args:
        src (ndarray): source image
        y_range (tuple of 2): location range in y-dimension
        x_range (tuple of 2): location range in x-dimension
    """
    result = np.zeros_like(src)
    return result


if __name__ == '__main__':
    """
    list your develop log or experiments for tone mapping here
    """
    print('tone mapping')
    '''
    radiance = cv.imread('../TestImg/memorial.hdr', -1)
    golden = cv.imread('../ref/p2_gtm.png')
    ldr = globalTM(radiance, scale=1.0)
    psnr = cv.PSNR(golden, ldr)

    impulse = np.load('../ref/p3_impulse.npy')
    golden = np.load('../ref/p3_gaussian.npy').astype(float)
    test = gaussianFilter(impulse, 5, 15).astype(float)
    psnr = cv.PSNR(golden, test)
    '''
    radiance = cv.imread('../TestImg/vinesunset.hdr', -1)
    golden = cv.imread('../ref/p3_ltm.png')
    gauhw1 = partial(gaussianFilter, N=35, sigma_s=100)
    test = localTM(radiance, gauhw1, scale=3)
    psnr = cv.PSNR(golden, test)
    