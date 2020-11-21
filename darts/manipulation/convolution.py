import numpy as np
import cv2 as cv

def convolution(frame, kernel, i, j, r_i, r_j):
    """
    apply convolution kernel to frame
    """
    frame = cv.copyMakeBorder(frame, r_i, r_i, r_j, r_j, cv.BORDER_REPLICATE)
    
    for y in range(frame.rows):
        for x in range(frame.cols):
            frame[y][x] = convolve(frame, i, j, r_i, r_j)

    return frame


def convolve(frame, kernel, i, j, r_i, r_j):
    """
    convolve (i,j) in frame with kernel
    """
    frame = cv.copyMakeBorder(frame, r_i, r_i, r_j, r_j, cv.BORDER_REPLICATE)
    sum = 0.0

    for m in range(-r_j, r_j):
        for n in range(-r_i, r_i):
            x, y, k_x, k_y = correct_indices
            frame_val = frame[y][x]
            kernel_val = kernel[k_y][k_x]
            sum += frame_val * kernel_val

    return sum

def correct_indices(i, j, m, n, r_i, r_j):
    """
    Correct frame and kernel indices for convolution
    """
    x = i + m + r_i
    y = j + n + r_j
    k_x = m + r_i
    k_y = n + r_j
    
    return x, y, k_x, k_y