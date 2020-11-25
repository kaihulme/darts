import cv2 as cv
import numpy as np
from darts.io.read import getpath
from darts.manipulation.utils import normalise, radtodeg

def write(frame, name):
    """
    Normalise frame then write to /out with given name.
    """
    frame = normalise(frame)
    path = getpath(name, "out")
    cv.imwrite(path, frame)

def gaussian(gaussian, name):
    """
    Write gaussian blurred frame to file.
    """
    write(gaussian, name + "_gaussian")

def sobel(sobel, name):
    """
    Write sobel components to file.
    """
    write(sobel.dfdx, name + "_dfdx")
    write(sobel.dfdy, name + "_dfdy")
    write(sobel.magnitude, name + "_magnitude")
    write(radtodeg(sobel.direction), name + "_direction")
    write(sobel.t_magnitude, name + "_magnitude_threshold")

def houghlines(houghlines, name):
    """
    Write hough lines to file. 
    """
    write(houghlines.hough_space, name + "_houghlines")
    write(houghlines.t_hough_space, name + "_houghlines_threshold")

def houghcircles(houghcircles, name, all=False):
    """
    Write hough spaces to file. 
    """
    if (all):
        for s, (space, t_space) in enumerate(zip(houghcircles.hough_space, houghcircles.t_hough_space)):
            write(space, name + "_houghcircles_" + str((s + 1) * houghcircles.r_size))
            write(t_space, name + "_houghcircles_" + str((s + 1) * houghcircles.r_size) + "_threshold")
    write(houghcircles.hough_space_sum, name + "_houghcircles_sum")
    write(houghcircles.t_hough_space_sum, name + "_houghcircles_sum_threshold")