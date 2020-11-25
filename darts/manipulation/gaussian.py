import cv2 as cv
from darts.manipulation.convolution import Convolution

class Gaussian():
    def __init__(self, size):
        self.kX = cv.getGaussianKernel(size, -1)
        self.kY = cv.transpose(self.kX)
        # self.kernel = self.kX * cv.transpose(self.kY)
        self.r = int((size - 1) / 2)
        self.size = size

    def blur(self, frame):
        """
        Apply gaussian kernel of specified size to (grey) frame.
        """
        # frame = convolution(frame, self.kX, 0, self.r)
        # frame = convolution(frame, self.kY, self.r, 0)
        frame = cv.GaussianBlur(frame, (self.size, self.size), 0) # faster
        return frame