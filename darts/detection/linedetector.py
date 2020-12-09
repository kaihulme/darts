import numpy as np
from darts.manipulation.utils import localmaxima

class LineDetector():
    def __init__(self, houghlines):
        self.hough_space = houghlines.hough_space
        self.t_hough_space = houghlines.t_hough_space
        self.max_rho = houghlines.max_rho
        self.max_theta = houghlines.max_theta
        self.height = houghlines.mag_height
        self.width = houghlines.mag_width
        self.lines = []

    def detect(self, min_dist=20):
        # find local maxima in thresholded hough space
        polar_lines = localmaxima(self.t_hough_space, min_dist)
        polar_lines[:, 0] -= self.max_rho
        polar_lines[:, 1] -= self.max_theta
        self.lines = [(getlinepoints(rho, theta, self.height, self.width)) for (rho, theta) in polar_lines]

# get line start and end points for drawing
def getlinepoints(rho, theta, height, width):
    a = np.cos(np.deg2rad(theta))
    b = np.sin(np.deg2rad(theta))
    x0 = (a * rho)
    y0 = (b * rho)
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return x1, y1, x2, y2