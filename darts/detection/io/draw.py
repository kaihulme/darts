import cv2 as cv
import numpy as np
import darts.io.write as write

def circles(frame, circles, name):
    """
    Draw circles on frame.
    """
    frame_copy = np.copy(frame)
    for (r, y, x) in circles:
        frame_copy = cv.circle(frame_copy, (x, y), r, (0, 255, 0), 2)
    write.circles(frame_copy, name)
    