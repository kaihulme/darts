import numpy as np
from darts.manipulation.utils import localmaxima

class LineDetector():
    def __init__(self, houghlines):
        self.hough_space = houghlines.hough_space
        self.t_hough_space = houghlines.t_hough_space
        self.lines = []

    def detect(self, min_dist=20):
        # find local maxima in thresholded hough space
        self.lines = localmaxima(self.t_hough_space, min_dist)
        # convert each polar line to cartesian
        # self.lines = np.asarray([((rho, theta), polartocartesian(rho, theta)) for (rho, theta) in polar_lines])

        

        print()

        # print(f"\n{len(self.lines)} lines found: {self.lines}")
        # print(f"\n{self.lines.shape}")

# polar to cartesian coordinates
def polartocartesian(rho, theta):
    x = rho * np.cos(np.deg2rad(theta))
    y = rho * np.sin(np.deg2rad(theta))
    return x, y