import cv2 as cv
from darts.manipulation.convolution import convolution

class Gaussian():
    def __init__(self, size):
        self.kX = cv.getGaussianKernel(size, -1)
        self.kY = cv.getGaussianKernel(size, -1)
        self.kernel = self.kX * cv.transpose(self.kY)
        self.r_i = int((size - 1) / 2)
        self.r_j = int((size - 1) / 2)
        # self.r_j = 1

    # TODO change gaussian to kX -> ky
    def blur(self, frame):
        """
        Apply gaussian kernel of specified size to (grey) frame
        """
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_blur = convolution(frame_gray, self.kernel, self.r_i, self.r_j)
        # frame_gray = convolution.convolution(frame_gray, self.kX, self.r_i, self.r_j)
        # frame_gray = convolution.convolution(frame_gray, self.kY, self.r_i, self.r_j)
        return frame_gray