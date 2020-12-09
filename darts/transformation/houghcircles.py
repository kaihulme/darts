import math
import numpy as np
from tqdm import tqdm
from darts.manipulation.utils import threshold

class HoughCircles():
    def __init__(self, min_r=40, max_r=100, r_step=5):
        self.min_r = min_r
        self.max_r = max_r
        self.r_step = r_step
        self.r_size = int((max_r-min_r) / r_step + 1)
        self.hough_space = []
        self.t_hough_space = []
        self.hough_space_sum = []
        self.t_hough_space_sum = []

    # TRY AND INCREMENT ALL HOUGHSPACE AT ONCE (np.add.at for 3D array)
    def transform(self, mag, dir, dt_size=2):
        """
        Apply Hough circles transformation to frame for a range of radii
        """

        # create r houghspaces for each radii r
        rows, cols = mag.shape
        self.hough_space = np.zeros((self.r_size, rows, cols))

        # precompute sin(theta), cos(theta) for all theta
        max_theta = 180
        thetas = np.arange(-max_theta, max_theta) * math.pi / 180
        cossin = np.column_stack((np.sin(thetas), np.cos(thetas)))

        # get positions of non-zero magnitude pixels and set initial radius
        points = np.column_stack(np.nonzero(mag))
        radius = self.min_r
        
        # progress bar
        pbar = tqdm(total=((len(points)) * (self.r_size)))
        for r in range(self.r_size):
            for (y, x) in points:
                dir_theta = np.rad2deg(dir[y][x]).astype('int')
                dt = (dir_theta - dt_size + max_theta, dir_theta + dt_size + max_theta)

                # compute circle points (y0, x0): y0=r-ysin(theta), x0=x-rcos(theta)
                c_points = np.column_stack((y - cossin[dt[0]:dt[1], 0] * radius, 
                                            x - cossin[dt[0]:dt[1], 1] * radius)).astype('int')
                # remove circle points that do not fit in frame
                c_points = c_points[np.where(~(c_points < 0).any(1) 
                                            & (c_points[:, 0] < rows) 
                                            & (c_points[:, 1] < cols))]
                # increment circle points in frame
                np.add.at(self.hough_space[r], (c_points[:, 0], c_points[:, 1]), 1)
                # update progress bar
                pbar.update(1)
            # increment radius
            radius += self.r_step

    def sum(self):
        """
        Sum hough space circles into one summed Hough space.
        """
        self.hough_space_sum = np.sum(self.t_hough_space, axis=0)

    def threshold(self, threshold_val):
        """
        Threshold each hough space, as well as the sum of Hough spaces.
        """
        self.t_hough_space = np.zeros(self.hough_space.shape)
        for r, space in enumerate(self.hough_space):
            self.t_hough_space[r] = threshold(space, threshold_val)

        # self.t_hough_space_sum = threshold(self.hough_space_sum, sum_threshold_val)