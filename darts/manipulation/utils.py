import os
import math
import cv2 as cv
import numpy as np

# faster with numpy: result=(frame>threshold)*frame, or frame[frame<threshold]=0
def threshold(frame, threshold):
    """
    Set values in frame below threshold to 0.
    """
    rows, cols = frame.shape
    frame_copy = np.zeros((rows, cols), dtype=float)
    for y in range(rows):
        for x in range(cols):
            if (frame[y][x] > float(threshold)): frame_copy[y][x] = frame[y][x]
    return frame_copy

# make same change as above
def threshold_abs(frame, threshold):
    """
    Set values in frame below threshold to 0 and above to 255.
    """
    rows, cols = frame.shape
    frame_copy = np.zeros((rows, cols), dtype=float)
    for y in range(rows):
        for x in range(cols):
            if (frame[y][x] > float(threshold)): frame_copy[y][x] = 255
    return frame_copy

# sklearn.preprocessing.normalise()
def normalise(frame):
    """
    MinMax normalisation of frame between 0-255
    """
    min, max, _, _ = cv.minMaxLoc(frame)
    rows, cols = frame.shape
    for y in range(rows):
        for x in range(cols):
            val = frame[y][x]
            if (max-min > 0): frame[y][x] = 255 * (val-min)/(max-min)
            else: frame[y][x] = 0
    return frame

def normalisewrite(frame, name):
    """
    Normalise frame then write to /out with given name.
    """
    frame = normalise(frame)
    path = getpath(name, "out")
    cv.imwrite(path, frame)

# np.rad2deg()
def radtodeg(frame):
    """
    Convert a matrix of radian angles to degrees.
    """
    pi = math.pi
    rows, cols = frame.shape
    for y in range(rows):
        for x in range(cols):
            rad = frame[y][x]
            deg = (rad if rad >= 0 else 2*pi + rad) * 360 / 2*pi
            frame[y][x] = deg
    return frame

def getpath(name, loc):
    """
    Get file path given name and folder name.
    """
    dir = os.getcwd()
    if (loc == "test") : return dir + "/darts/resources/images/test/" + name + ".png"
    elif (loc == "out") : return dir + "/darts/out/" + name + ".png"
    else: return False