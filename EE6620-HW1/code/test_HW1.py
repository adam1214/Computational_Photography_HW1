import unittest
import numpy as np
from functools import partial
import cv2 as cv
from cr_calibration import estimateResponse, constructRadiance, loadExposures
from tm import globalTM, localTM, gaussianFilter, bilateralFilter, whiteBalance


class TestHW1(unittest.TestCase):

    def test_estimateResponse(self):
        samples = np.load('../ref/p1_pixel_samples.npy')
        etime = np.load('../ref/p1_et_samples.npy')
        golden = np.load('../ref/p1_resp.npy')
        resp_test = estimateResponse(samples, etime)
        mse = np.mean((golden - resp_test)**2)
        self.assertLessEqual(mse, 0.1)
        return mse

    def test_constructRadiance(self):
        golden = np.load('../ref/p1_rad.npy')
        cimg_list = np.load('../ref/p1_cimg.npy')
        etime = np.load('../ref/p1_et_samples.npy')
        resp = np.load('../ref/p1_resp.npy')
        rad_test = constructRadiance(cimg_list, resp, etime)
        mse = np.mean((golden - rad_test)**2)
        self.assertLessEqual(mse, 0.01)
        return mse

    def test_globalTM(self):
        radiance = cv.imread('../TestImg/memorial.hdr', -1)
        golden = cv.imread('../ref/p2_gtm.png')
        ldr = globalTM(radiance, scale=1.0)
        psnr = cv.PSNR(golden, ldr)
        self.assertGreaterEqual(psnr, 45)
        return psnr

    def test_localTMgaussian(self):
        radiance = cv.imread('../TestImg/vinesunset.hdr', -1)
        golden = cv.imread('../ref/p3_ltm.png')
        gauhw1 = partial(gaussianFilter, N=35, sigma_s=100)
        test = localTM(radiance, gauhw1, scale=3)
        psnr = cv.PSNR(golden, test)
        self.assertGreaterEqual(psnr, 45)
        return psnr

    def test_gaussian(self):
        impulse = np.load('../ref/p3_impulse.npy')
        golden = np.load('../ref/p3_gaussian.npy').astype(float)
        test = gaussianFilter(impulse, 5, 15).astype(float)
        psnr = cv.PSNR(golden, test)
        self.assertGreaterEqual(psnr, 60)
        return psnr

    def test_localTMbilateral(self):
        radiance = np.load('../ref/p4_imgpatch.npy')
        golden = cv.imread('../ref/p4_ltm_patch.png')
        bilhw1 = partial(bilateralFilter, N=35, sigma_s=100, sigma_r=0.8)
        test = localTM(radiance, bilhw1, scale=3)
        psnr = cv.PSNR(golden, test)
        self.assertGreaterEqual(psnr, 45)
        return psnr

    def test_bilateral(self):
        step = np.load('../ref/p4_step.npy')
        golden = np.load('../ref/p4_bilateral.npy').astype(float)
        test = bilateralFilter(step, 9, 50, 10).astype(float)
        psnr = cv.PSNR(golden, test)
        self.assertGreaterEqual(psnr, 60)
        return psnr

    def test_whiteBalance(self):
        img = np.random.rand(30, 30, 3)
        ktbw = (slice(0, 15), slice(0, 15))
        w_avg = img[0:15, 0:15, 2].mean()
        wb_result = whiteBalance(img, (0, 15), (0, 15))
        result_avg = wb_result[ktbw].mean(axis=(0, 1))
        self.assertAlmostEqual(result_avg[0], w_avg)
        self.assertAlmostEqual(result_avg[1], w_avg)

    def test_globalTMwb(self):
        radiance = cv.imread('../TestImg/memorial.hdr', -1)
        golden = cv.imread('../ref/p5_wb_gtm.png')
        wb_hdr = whiteBalance(radiance, (457, 481), (400, 412))
        test = globalTM(wb_hdr)
        psnr = cv.PSNR(golden, test)
        self.assertGreaterEqual(psnr, 45)
        return psnr


if __name__ == '__main__':
    unittest.main()
