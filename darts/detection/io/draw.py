import cv2 as cv
import numpy as np
import darts.io.write as write

def lines(frame, lines, name):
    """
    Draw circles on frame.
    """
    rows, cols, _ = frame.shape
    frame_copy = np.copy(frame)

    for (rho, theta) in lines:

        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + cols / 2
        y0 = (b * rho) + rows / 2
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        print(f"(x1, y1), (x2, y2): ({x1}, {y1}), ({x2}, {y2})")
        frame_copy = cv.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    write.lines(frame_copy, name)

    # for ((rho, theta), (a, b)) in lines:
        # x0 = (a * rho) + (cols / 2)
        # y0 = (b * rho) + (rows / 2)
        # print(f"(x0, y0): ({x0}, {y0})")
        # x1 = int(x0 + 1000 * (-b))
        # y1 = int(y0 + 1000 * (a))
        # x2 = int(x0 - 1000 * (-b))
        # y2 = int(y0 - 1000 * (a))
        # frame_copy = cv.line(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # print(f"(x1, y1), (x2, y2): ({x1}, {y1}), ({x2}, {y2})")

def circles(frame, circles, name):
    """
    Draw circles on frame.
    """
    frame_copy = np.copy(frame)
    for (r, y, x) in circles:
        frame_copy = cv.circle(frame_copy, (x, y), r, (0, 255, 0), 2)
    write.circles(frame_copy, name)
    