import cv2 as cv
import numpy as np
from darts.io.read import getpath
from darts.tools.utils import normalise

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
    write(np.rad2deg(sobel.direction), name + "_direction")
    write(sobel.t_magnitude, name + "_magnitude_threshold")

def houghlines(houghlines, name):
    """
    Write hough lines to file. 
    """
    write(houghlines.hough_space, name + "_houghlines")
    write(houghlines.t_hough_space, name + "_houghlines_threshold")

def lines(frame, name):
    """
    Write image with circles
    """
    write(frame, name + "_lines")

def houghcircles(houghcircles, name, all=False):
    """
    Write hough spaces to file. 
    """
    if (all == True):
        for s, (space, t_space) in enumerate(zip(houghcircles.hough_space, houghcircles.t_hough_space)):
            write(space, name + "_houghcircles_" + str(s + 1))
            write(t_space, name + "_houghcircles_" + str(s + 1) + "_threshold")
    write(houghcircles.hough_space_sum, name + "_houghcircles_sum")

def circles(frame, name):
    """
    Write image with circles
    """
    write(frame, name + "_circles")

def true_face_boxes(frame, name):
    """
    Write image with ground truth bounding boxes for faces
    """
    write(frame, name + "_true_faces")

def true_dart_boxes(frame, name):
    """
    Write image with ground truth bounding boxes for dartboards
    """
    write(frame, name + "_true_dartboards")

def face_boxes(frame, name):
    """
    Write image with Viola Jones face detection bounding boxes
    """
    write(frame, name + "_vj_faces")

def dart_boxes(frame, name):
    """
    Write image with Viola Jones dartboard detection bounding boxes
    """
    write(frame, name + "_vj_dartboards")

def ensemble_boxes(frame, name):
    """
    Write image with ensemble dartboard detection bounding boxes
    """
    write(frame, name + "_ensemble_dartboards")

def true_pred_face_boxes(frame, name):
    """
    Write image with ground truth bounding boxes
    and predicted bounding boxes for vj faces
    """
    write(frame, name + "_true_pred_vj_face")

def true_pred_dart_boxes(frame, name):
    """
    Write image with ground truth bounding boxes
    and predicted bounding boxes for vj dartboards
    """
    write(frame, name + "_true_pred_vj_dartboards")

def true_pred_ensemble_boxes(frame, name):
    """
    Write image with ground truth bounding boxes
    and predicted bounding boxes for ensemble
    """
    write(frame, name + "_true_pred_ensemble_dartboards")

def clustered(frame, name):
    """
    Write clustered image
    """
    write(frame, name + "_clustered")

def canny(frame, name):
    """
    Write canny image
    """
    write(frame, name + "_canny")

def clustered_canny(frame, name):
    """
    Write canny clustered image
    """
    write(frame, name + "_clustered_canny")

def contour(frame, name):
    """
    Write countoured image
    """
    write(frame, name + "_countour")