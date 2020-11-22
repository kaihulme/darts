import cv2 as cv
from darts.manipulation.convolution import convolution

class Gaussian():
    def __init__(self, size):
        self.kernel = cv.getGaussianKernel(size, -1)
        self.r_i = 0
        self.r_j = int((size-1)/2)
        # self.kX = cv.getGaussianKernel(size, -1)
        # self.kY = cv.getGaussianKernel(size, -1)
        # self.kernel = self.kX * cv.transpose(self.kY)
        # self.r_i = int((size - 1) / 2)
        # self.r_j = int((size - 1) / 2)

    def blur(self, frame):
        """
        Apply gaussian kernel of specified size to (grey) frame
        """
        frame = convolution(frame, self.kernel, self.r_i, self.r_j)
        return frame
        # frame_gray = convolution(frame_gray, self.kX, self.r_i, self.r_j)
        # frame_gray = convolution(frame_gray, self.kY, self.r_i, self.r_j)