from darts.io.draw import circles
import numpy as np
import scipy.ndimage.filters as filters
from skimage.feature import peak_local_max

class CircleDetector():
    def __init__(self, houghcircles):
        self.hough_space = houghcircles.hough_space
        self.t_hough_space = houghcircles.t_hough_space
        self.t_hough_space_sum = houghcircles.t_hough_space_sum
        self.min_r = houghcircles.min_r
        self.r_step = houghcircles.r_step
        self.circles = []

    def detect(self, min_dist=20):
        # find local maxima in thresholded hough space sum
        centres = localmaxima(self.t_hough_space_sum, min_dist)
        # find radius for circle as max r 
        for (y, x) in centres:
            radius_i = np.argmax(self.hough_space[:][:, y][:, x])
            radius = self.min_r + radius_i * self.r_step
            circle = np.array([radius, y, x])
            self.circles.append(circle)
        print(f"\n{len(self.circles)} circles found: {self.circles}")


def localmaxima(frame, min_dist):
    centres = peak_local_max(frame, min_dist)
    return centres