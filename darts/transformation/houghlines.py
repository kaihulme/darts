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

    def transform(self, frame):
        """
        Apply Hough circles transformation to frame for a range of radii
        """
        # create hough space
        rows, cols = frame.shape
        max_rho = int(math.hypot(rows, cols))
        max_theta = 180
        self.hough_space = np.zeros((max_rho*2, max_theta*2))

        # precompute sin(theta), cos(theta) for all theta
        thetas = np.arange(-max_theta, max_theta+1, self.t_step)
        cossin = np.column_stack((np.sin(thetas * math.pi / 180), 
                                  np.cos(thetas * math.pi / 180)))

        # get positions of non-zero magnitude pixels and set inital radius
        points = np.column_stack(np.nonzero(frame))

        for (y, x) in points:

            # caluclate rho = ysin(theta) + xcos(theta)
            rhos = y * cossin[:, 0] + x * cossin[:, 1]

            # set line points as (rho, theta) and remove points not in space size
            l_points = np.column_stack((rhos, thetas)).astype('int')
            l_points = l_points[np.where((l_points[:, 0] >= -max_rho) 
                                        & (l_points[:, 1] >= -max_theta) 
                                        & (l_points[:, 0] < max_rho) 
                                        & (l_points[:, 1] < max_theta))]

            for (rho, theta) in l_points:
                self.hough_space[rho+max_rho][theta+max_theta] += 1
                # print(f"(theta, rho): ({theta}, {rho})")
                # if (theta >= 0 and rho >= 0 and theta < rows and rho < cols):

    def threshold(self, threshold_val):
        """
        Threshold each hough space, as well as the sum of Hough spaces.
        """
        # for r, space in enumerate(self.hough_space):
            # self.hough_space[r] = threshold(space, threshold_val)
        self.t_hough_space = threshold(self.hough_space, threshold_val)
        # self.t_hough_space_sum = threshold(self.hough_space_sum, threshold_val)