import numpy as np
import cv2 as cv

def convolution(frame, kernel, r_i, r_j, d=2):
    """
    apply convolution kernel to frame
    """
    rows, cols = frame.shape
    frame_copy = cv.copyMakeBorder(frame, r_i, r_i, r_j, r_j, cv.BORDER_REPLICATE)
    
    for y in range(rows):
        for x in range(cols):
            frame[y][x] = convolve(frame_copy, kernel, x, y, r_i, r_j)

    return frame_copy


def convolve(frame, kernel, i, j, r_i, r_j):
    """
    convolve (i,j) in frame with kernel
    """
    sum = 0.0

    # if(r_j > 1):
    for m in range(-r_i, r_i+1):
        for n in range(-r_j, r_j+1):
            x, y, k_x, k_y = correct_indices(i, j, m, n, r_i, r_j)
            frame_val = frame[y][x]
            kernel_val = kernel[k_y][k_x]
            sum += frame_val * kernel_val
    # else:
        # for m in range(-r_j, r_j+1):
            # x, y, k_x = correct_indices(i, j, m, 0, r_i, r_j)
            # frame_val = frame[y][x]
            # kernel_val = kernel[k_x]
            # sum += frame_val * kernel_val

    return sum

def correct_indices(i, j, m, n, r_i, r_j):
    """
    Correct frame and kernel indices for convolution
    """
    x = i + m + r_i
    y = j + n + r_j
    k_x = m + r_i
    # if (r_j == 1 ): return x, y, k_x
    k_y = n + r_j
    return x, y, k_x, k_y