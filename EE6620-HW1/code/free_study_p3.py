import os
import cv2 as cv
import numpy as np
from cr_calibration import estimateResponse, constructRadiance, loadExposures, pixelSample

if __name__ == '__main__':
    lambda_ = 50
    
    img_list1, exposure_times1 = loadExposures('../TestImg/free_study_p3_set1')
    img_list2, exposure_times2 = loadExposures('../TestImg/free_study_p3_set2')
    # img_list1 or 2(16, 768, 512, 3)(uint8):4張照片，每張照片768*512，3 channels
    # exposure_times1 or 2(list):4張照片的曝光時間
    radiance1 = np.zeros_like(img_list1[0], dtype=np.float32)
    pixel_samples = pixelSample(img_list2) #(16, 96, 3)(uint8) 每64個pixel做一次採樣，三個channel都採樣一樣的pixel
    for ch in range(3):
        response = estimateResponse(pixel_samples[..., ch], exposure_times2, lambda_)
        radiance1[..., ch] = constructRadiance(img_list1[..., ch], response, exposure_times1)


    radiance2 = np.zeros_like(img_list1[0], dtype=np.float32)
    pixel_samples = pixelSample(img_list1) #(16, 96, 3)(uint8) 每64個pixel做一次採樣，三個channel都採樣一樣的pixel
    for ch in range(3):
        response = estimateResponse(pixel_samples[..., ch], exposure_times1, lambda_)
        radiance2[..., ch] = constructRadiance(img_list1[..., ch], response, exposure_times1)
    
    mse = np.mean((radiance1 - radiance2)**2)
    print(mse)
    
    #############################################################################################
    img_list, exposure_times = loadExposures('../TestImg/memorial')
    # img_list(16, 768, 512, 3)(uint8):16張照片，每張照片768*512，3 channels
    # exposure_times(list):16張照片的曝光時間
    radiance3 = np.zeros_like(img_list[0], dtype=np.float32)
    pixel_samples = pixelSample(img_list) #(16, 96, 3)(uint8) 每64個pixel做一次採樣，三個channel都採樣一樣的pixel
    for ch in range(3):
        response = estimateResponse(pixel_samples[..., ch], exposure_times, lambda_)
        radiance3[..., ch] = constructRadiance(img_list[..., ch], response, exposure_times)
    
    for i in range(0, img_list.shape[0], 1):
        # img_list(16, 768, 512, 3)(uint8):16張照片，每張照片768*512，3 channels
        # exposure_times(list):16張照片的曝光時間
        radiance4 = np.zeros_like(img_list[0], dtype=np.float32)
        pixel_samples = pixelSample(img_list) #(16, 96, 3)(uint8) 每64個pixel做一次採樣，三個channel都採樣一樣的pixel
        for ch in range(3):
            response = estimateResponse(pixel_samples[..., ch], exposure_times, lambda_)
            single_img_list = img_list[i, ...]
            single_img_list = single_img_list[np.newaxis, :]
            radiance4[..., ch] = constructRadiance(single_img_list[..., ch], response, exposure_times)

        mse = np.mean((radiance3 - radiance4)**2)
        print(mse)
