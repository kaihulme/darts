import cv2 as cv
import numpy as np
from darts.manipulation.convolution import convolution, convolve

class Sobel():
    def __init__(self):
        self.dfdx = []
        self.dfdy = []
        self.magnitude = []
        self.direction = []
        self.t_magnitude = []
        self._r_i = 1
        self._r_j = 1
        self._dfdx_kernel = [[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]]
        self._dfdy_kernel = [[-1,-1,-1],
                             [ 0, 0, 0],
                             [ 1, 1, 1]]

    def edgedetection(self, frame):
        """
        Apply sobel edge detection to frame
        """
        self.dfdx = convolution(frame, self._dfdx_kernel, self._r_i, self._r_j)
        self.dfdy = convolution(frame, self._dfdy_kernel, self._r_i, self._r_j)
        
        # for y in range(rows-1):
        #     for x in range(cols-1):
        #         self.dfdx[y][x] = convolve(frame_copy, self._dfdx_kernel, x, y, self._r_i, self._r_j)
        #         self.dfdy[y][x] = convolve(frame_copy, self._dfdy_kernel, x, y, self._r_i, self._r_j)

    # def thresholdmagnitude(self, frame, threshold):
        # """
        # Threshold the gradient magnitudes produced sobel
        # """