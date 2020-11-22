import cv2 as cv
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
    write(sobel.t_magnitude, name + "_threshold_magnitude")

def houghcircles(houghcircles, name, all=False):
    """
    Write hough spaces to file. 
    """
    if (all):
        for s, (space, t_space) in enumerate(zip(houghcircles.hough_space, houghcircles.t_hough_space)):
            write(space, name + "_houghspace_" + str((s + 1) * houghcircles.r_size))
            write(t_space, name + "_tresholded_houghspace_" + str((s + 1) * houghcircles.r_size))
    write(houghcircles.hough_space_sum, name + "_houghspace_summed")
    write(houghcircles.t_hough_space_sum, name + "_thresholded_houghspace_summed")