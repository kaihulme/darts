import math
import cv2 as cv
import numpy as np
from darts.manipulation.utils import threshold

class HoughLines():
    # def __init__(self, min_r=10, max_r=100, r_step=1, t_step=1):
    def __init__(self, t_step=1):
        self.t_step = t_step
        self.hough_space = []
        self.t_hough_space = []

    # very slow needs improving!
    def transform(self, frame):
        """
        Apply Hough circles transformation to frame for a range of radii
        """
        rows, cols = frame.shape
        thetas = [theta for theta in range (0, 360, self.t_step)]

        self.hough_space = np.zeros((rows, cols))
        points = np.nonzero(frame)

        # for r in range(self.r_size):
        for (y, x) in zip(points[0], points[1]):
            x0s = np.array(x * np.cos(thetas)).astype('int')
            y0s = np.array(y * np.sin(thetas)).astype('int')
            rhos = x0s + y0s
            for (rho, theta) in zip(rhos, thetas):
                if (rho >= 0 and theta >= 0 and rho < cols and theta < rows):
                    self.hough_space[theta][rho] += 1

    def threshold(self, threshold_val):
        """
        Threshold each hough space, as well as the sum of Hough spaces.
        """
        # for r, space in enumerate(self.hough_space):
            # self.hough_space[r] = threshold(space, threshold_val)
        self.t_hough_space = threshold(self.hough_space, threshold_val)
        # self.t_hough_space_sum = threshold(self.hough_space_sum, threshold_val)