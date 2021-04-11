import time
from functools import partial
import cv2 as cv
import numpy as np
import math
import sys

from cr_calibration import wholeFlow
from tm import globalTM, localTM, gaussianFilter, bilateralFilter, whiteBalance


def globalTM_edit(src, scale=1.0):
    """Global tone mapping for experiment c

    Args:
        src (ndarray, float32): source radiance image
        scale (float, optional): scaling factor (Defaults to 1.0)
    """
    result = np.zeros_like(src, dtype=np.float32)
    
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
                    adjustment = scale*(sys.float_info.min - math.log(max_rad[channel],2))+math.log(max_rad[channel],2)
                compressed_result = math.pow(2, adjustment)
                gamma_correction = math.pow(compressed_result, 1/gamma)
                '''
                gamma_correction = gamma_correction * 255
                if gamma_correction > 255:
                    gamma_correction = 255
                elif gamma_correction < 0:
                    gamma_correction = 0
                result[i][j][channel] = gamma_correction
                '''
                result[i][j][channel] = gamma_correction
    
    # Min-Max Normalization -> 乘255
    result_0_max = np.max(result[..., 0]) #scalar
    result_0_min = np.min(result[..., 0]) #scalar

    result_1_max = np.max(result[..., 1]) #scalar
    result_1_min = np.min(result[..., 1]) #scalar

    result_2_max = np.max(result[..., 2]) #scalar
    result_2_min = np.min(result[..., 2]) #scalar
    for channel in range(0, result.shape[2], 1):
        for i in range(0, result.shape[0], 1):
            for j in range(0, result.shape[1], 1):
                if channel == 0:
                    result[i][j][channel] = (result[i][j][channel] - result_0_min)/(result_0_max - result_0_min)
                elif channel == 1:
                    result[i][j][channel] = (result[i][j][channel] - result_1_min)/(result_1_max - result_1_min)
                elif channel == 2:
                    result[i][j][channel] = (result[i][j][channel] - result_2_min)/(result_2_max - result_2_min)

                result[i][j][channel] = result[i][j][channel] * 255
                if result[i][j][channel] > 255:
                    result[i][j][channel] = 255
                elif result[i][j][channel] < 0:
                    result[i][j][channel] = 0
    return result.astype('uint8')

if __name__ == '__main__':
    
    gamma = 2.2
    # Demonstrate Overall Flow of HDR Imaging

    # Declare results
    radiance = None
    radiance_wb = None
    gtm = None
    ltm = None
    ltm_edge = None

    # resize to 768*512 and save as png format
    image1 = cv.imread('../TestImg/experiment_c/IMG_4782.JPG')
    image2 = cv.imread('../TestImg/experiment_c/IMG_4783.JPG')
    image3 = cv.imread('../TestImg/experiment_c/IMG_4784.JPG')

    image1_resize = cv.resize(image1, (512, 768), interpolation=cv.INTER_AREA)
    image2_resize = cv.resize(image2, (512, 768), interpolation=cv.INTER_AREA)
    image3_resize = cv.resize(image3, (512, 768), interpolation=cv.INTER_AREA)

    cv.imwrite('../TestImg/experiment_c/IMG_4782_resize.png', image1_resize)
    cv.imwrite('../TestImg/experiment_c/IMG_4783_resize.png', image2_resize)
    cv.imwrite('../TestImg/experiment_c/IMG_4784_resize.png', image3_resize)

    # 1-1 Camera response calibration
    radiance = wholeFlow('../TestImg/experiment_c', lambda_=50)

    # 1-5 White Balance
    if radiance is not None:
        ktbw = (0, 118), (0, 74) # 人工搜尋，找image3_resize(最暗的拍攝圖)中RGB值最高的pixel位置(最接近白色)
        radiance_wb = whiteBalance(radiance, *ktbw)

    # 1-2 Global tone mapping
    if radiance_wb is not None:
        gtm = globalTM_edit(radiance_wb)
        cv.imwrite('../result/Experiments/c/gtm.png', gtm)

    # 1-3 Local tone mapping with Gaussian
    if radiance_wb is not None:
        gauhw1 = partial(gaussianFilter, N=15, sigma_s=100)
        ltm = localTM(radiance_wb, gauhw1, scale=4)
        cv.imwrite('../result/Experiments/c/ltm_gaussian.png', ltm)

    # 1-4 Edge-Preserving filter
    # Note that bilateral filter may be slow for large window size.
    if radiance_wb is not None:
        bilhw1 = partial(bilateralFilter, N=15, sigma_s=100, sigma_r=0.8)
        ltm_edge = localTM(radiance_wb, bilhw1, scale=4)
        cv.imwrite('../result/Experiments/c/ltm_bilateral.png', ltm_edge)
    
