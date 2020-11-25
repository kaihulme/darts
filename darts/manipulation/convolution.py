import cv2 as cv
import numpy as np

class Convolution():
    def __init__(self, frame, kernel, r_i, r_j):
        """
        Convolution class for faster convolutions
        """
        rows, cols = frame.shape
        self.frame = cv.copyMakeBorder(frame, r_j, r_j, r_i, r_i, cv.BORDER_REPLICATE)
        self.rows = rows
        self.cols = cols
        self.kernel = kernel
        self.r_i = r_i
        self.r_j = r_j        

    def convolveframe(self):
        """
        apply convolution kernel to frame
        """
        # rows, cols = self.frame.shape
        # frame_copy = cv.copyMakeBorder(self.frame, self.r_j, self.r_j, self.r_i, self.r_i, cv.BORDER_REPLICATE)
        result = np.zeros((self.rows, self.cols), dtype=float)
        for y in range(self.rows):
            for x in range(self.cols):
                result[y][x] = self.convolve(x, y)
        return result

    def convolve(self, i, j):
        """
        convolve (i,j) in frame with kernel
        """
        sum = 0.0
        for m in range(-self.r_i, self.r_i+1):
            for n in range(-self.r_j, self.r_j+1):
                x, y, k_x, k_y = correct_indices(i, j, m, n, self.r_i, self.r_j)
                frame_val = self.frame[y][x]
                kernel_val = self.kernel[k_y][k_x]
                sum += frame_val * kernel_val
        return sum

def convolution(frame, kernel, r_i, r_j):
    """
    apply convolution kernel to frame
    """
    rows, cols = frame.shape
    frame_copy = cv.copyMakeBorder(frame, r_j, r_j, r_i, r_i, cv.BORDER_REPLICATE)
    out = np.zeros((rows, cols), dtype=float)
    for y in range(rows):
        for x in range(cols):
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