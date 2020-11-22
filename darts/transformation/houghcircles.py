import math
import cv2 as cv
import numpy as np

class HoughCircles():
    def __init__(self, min_r=10, max_r=100, r_step=1, t_step=1):
        self.min_r = min_r
        self.max_r = max_r
        self.r_step = r_step
        self.t_step = t_step
        self.r_size = int((max_r-min_r) / r_step + 1)
        self.hough_space = []
        self.t_hough_space = []
        self.hough_space_sum = []
        self.t_hough_space_sum = []

    def transform(self, frame):
        """
        Apply hough circles transformation to frame for a range of radii
        """
        pi = math.pi
        rows, cols = frame.shape
        radius = self.min_r
        self.hough_space = np.zeros((self.r_size, rows, cols))
        thetas = [(math.cos(theta*pi/180), math.sin(theta*pi/180)) for theta in range(0, 360, self.t_step)]        
        for r in range(self.r_size):
            self.hough_space[r] = np.zeros((rows, cols))
            for y in range(rows):
                for x in range(cols):
                    val = frame[y][x]
                    if (val > 0):
                        for theta in thetas:
                            x0 = int(x - radius*theta[0])
                            y0 = int(y - radius*theta[1])
                            if (x0>=0 and y0>=0 and x0<cols and y0<rows):
                                self.hough_space[r][y0][x0] += 1
            radius += self.r_step