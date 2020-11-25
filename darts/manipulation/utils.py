import os
import math
import cv2 as cv
import numpy as np

# faster with numpy: result=(frame>threshold)*frame, or frame[frame<threshold]=0
def threshold(frame, threshold):
    """
    Set values in frame below threshold to 0.
    """
    frame_copy = np.copy(frame)
    frame_copy[frame_copy < threshold] = 0
    return frame_copy

# make similar change to above
def threshold_abs(frame, threshold):
    """
    Set values in frame below threshold to 0 and above to 255.
    """
    frame_copy = np.copy(frame)
    frame_copy[frame_copy < threshold] = 0
    frame_copy[frame_copy >= threshold] = 255
    return frame_copy

# sklearn.preprocessing.normalise()
def normalise(frame):
    """
    MinMax normalisation of frame between 0-255
    """
    # MIGHT NOT BE WORKING
    frame_copy = np.copy(frame)
    min = np.min(frame_copy)
    max = np.max(frame_copy)
    frame_copy = (frame - min) / (max - min) * 255
    return frame_copy
    # min, max, _, _ = cv.minMaxLoc(frame)
    # rows, cols = frame.shape
    # for y in range(rows):
        # for x in range(cols):
            # val = frame[y][x]
            # if (max-min > 0): frame[y][x] = 255 * (val-min)/(max-min)
            # else: frame[y][x] = 0
    # return frame

# np.rad2deg() is faster
def radtodeg(frame):
    """
    Convert a matrix of radian angles to degrees.
    """
    pi = math.pi
    rows, cols = frame.shape
    deg_frame = np.zeros((rows, cols))
    for y in range(rows):
        for x in range(cols):
            rad = frame[y][x]
            deg = (rad if rad >= 0 else 2*pi + rad) * 360 / 2*pi
            deg_frame[y][x] = deg
    return deg_frame