import math
import cv2 as cv
import numpy as np
from tm import whiteBalance, globalTM

def globalTM_pure(src, scale=1.0):
    """Global tone mapping (section 1-2)

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
                    adjustment = scale*(-10000000-math.log(max_rad[channel],2))+math.log(max_rad[channel],2)
                compressed_result = math.pow(2, adjustment)
                result[i][j][channel] = compressed_result
                '''
                gamma_correction = math.pow(compressed_result, 1/gamma)
                gamma_correction = gamma_correction * 255
                if gamma_correction > 255:
                    gamma_correction = 255
                elif gamma_correction < 0:
                    gamma_correction = 0
                result[i][j][channel] = gamma_correction
                '''
    return result

def gamma_correction_fun(compressed_result, gamma):
    gamma_correction = math.pow(compressed_result, 1/gamma)
    gamma_correction = gamma_correction * 255
    if gamma_correction > 255:
        gamma_correction = 255
    elif gamma_correction < 0:
        gamma_correction = 0
    return gamma_correction

radiance = cv.imread('../TestImg/memorial.hdr', -1)
golden = cv.imread('../ref/p5_wb_gtm.png')
wb_hdr = whiteBalance(radiance, (457, 481), (400, 412))
test = globalTM(wb_hdr)
psnr = cv.PSNR(golden, test)
cv.imwrite('../result/Free_Study/problem_8/p5_wb_gtm.png', test)
print('PSNR of whiteBalance -> global tone mapping:', psnr)

radiance = cv.imread('../TestImg/memorial.hdr', -1)
golden = cv.imread('../ref/p5_wb_gtm.png')
test = globalTM_pure(radiance)
wb_hdr = whiteBalance(test, (457, 481), (400, 412))
for channel in range(0, wb_hdr.shape[2], 1):
    for i in range(0, wb_hdr.shape[0], 1):
        for j in range(0, wb_hdr.shape[1], 1):
            wb_hdr[i][j][channel] = gamma_correction_fun(wb_hdr[i][j][channel], 2.2)
psnr = cv.PSNR(golden, wb_hdr.astype('uint8'))
cv.imwrite('../result/Free_Study/problem_8/p5_gtm_wb.png', wb_hdr)
print('PSNR of global tone mapping -> whiteBalance:', psnr)
