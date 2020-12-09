import math
import numpy as np
from tqdm import tqdm
from darts.manipulation.utils import threshold

class HoughLines():
    def __init__(self):
        self.hough_space = []
        self.t_hough_space = []
        self.mag_width = 0
        self.mag_height = 0
        self.max_rho = 0
        self.max_theta = 180

    def transform(self, mag, dir, dt_size=2):
        """
        Apply Hough circles transformation to mag for a range of radii
        """
        # create hough space with dimensions (-max ρ to max ρ, -max θ to max θ)
        # where max ρ is the diagonal length of the input magnitude
        # and max θ is (generally) 180°
        rows, cols = mag.shape
        self.mag_height = rows
        self.mag_width = cols
        self.max_rho = int(math.hypot(rows, cols))
        self.hough_space = np.zeros((self.max_rho*2, self.max_theta*2))
        # precompute sin(θ), cos(θ) for all θ
        thetas = np.arange(-self.max_theta, self.max_theta)
        cossin = np.column_stack((np.sin(thetas * math.pi / 180), np.cos(thetas * math.pi / 180)))
        # get positions of non-zero magnitude pixels
        points = np.column_stack(np.nonzero(mag))        
        # progress bar
        for (y, x) in tqdm(points):
            # Get range of 0: Δ0 = 0 from gradient direction ± the Δ0 size
            dir_theta = np.rad2deg(dir[y][x]).astype('int')
            dt = (dir_theta - dt_size + self.max_theta, dir_theta + dt_size + self.max_theta)    
            # calculate ρ = ysin(θ) + xcos(θ) for θ in Δ0
            rhos = y * cossin[dt[0]:dt[1], 0] + x * cossin[dt[0]:dt[1], 1]
            # set line points as each pair (ρ, θ) where θ is in Δ0 and remove points not in space size
            l_points = np.column_stack((rhos, thetas[dt[0]:dt[1]])).astype('int')
            l_points = l_points[np.where( (l_points[:, 0] >= -self.max_rho) 
                                        & (l_points[:, 1] >= -self.max_theta) 
                                        & (l_points[:, 0] < self.max_rho) 
                                        & (l_points[:, 1] < self.max_theta))]
            # shift line points to centre polar origin and increment each in hough space
            np.add.at(self.hough_space, (l_points[:, 0] + self.max_rho, l_points[:, 1] + self.max_theta), 1)


    def threshold(self, threshold_val):
        """
        Threshold each hough space, as well as the sum of Hough spaces.
        """
        self.t_hough_space = threshold(self.hough_space, threshold_val)