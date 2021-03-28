# Functions for tone mapping
# Put all your related functions for section 1-2~1-5 in this file.
import numpy as np
gamma = 2.2


def globalTM(src, scale=1.0):
    """Global tone mapping (section 1-2)

    Args:
        src (ndarray, float32): source radiance image
        scale (float, optional): scaling factor (Defaults to 1.0)
    """
    result = np.zeros_like(src, dtype=np.uint8)
    return result


def localTM(src, imgFilter, scale=3):
    """Local tone mapping (section 1-3)

    Args:
        src (ndarray, float32): source radiance image
        imgFilter (function): filter function with preset parameters
        scale (float, optional): scaling factor (Defaults to 3)
    """
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
    result = np.zeros_like(src, dtype=dtype)
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
