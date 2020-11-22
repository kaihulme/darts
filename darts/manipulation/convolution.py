import cv2 as cv
import numpy as np

def convolution(frame, kernel, r_i, r_j):
    """
    apply convolution kernel to frame
    """
    rows, cols = frame.shape
    frame_copy = cv.copyMakeBorder(frame, r_j, r_j, r_i, r_i, cv.BORDER_REPLICATE)
    out = np.zeros((rows,cols), dtype=float)
    for y in range(rows-1):
        for x in range(cols-1):
            out[y][x] = convolve(frame_copy, kernel, x, y, r_i, r_j)    
    return out


def convolve(frame, kernel, i, j, r_i, r_j):
    """
    convolve (i,j) in frame with kernel
    """
    sum = 0.0
    for m in range(-r_i, r_i+1):
        for n in range(-r_j, r_j+1):
            x, y, k_x, k_y = correct_indices(i, j, m, n, r_i, r_j)
            frame_val = frame[y][x]
            kernel_val = kernel[k_y][k_x]
            sum += frame_val * kernel_val
    return sum


def correct_indices(i, j, m, n, r_i, r_j):
    """
    Correct frame and kernel indices for convolution
    """
    if (r_i > 0): 
        i += m + r_i
        m += r_i
    if (r_j > 0):
        j += n + r_j
        n += r_j
    return i, j, m, n
