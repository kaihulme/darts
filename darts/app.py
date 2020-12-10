import os
import sys
import cv2 as cv
import darts.io.draw as draw
import darts.io.write as write
from darts.io.read import readfromargs
from darts.manipulation.gaussian import Gaussian
from darts.transformation.houghlines import HoughLines
from darts.transformation.houghcircles import HoughCircles
from darts.detection.edgedetector import Sobel
from darts.detection.violajones import ViolaJones
from darts.detection.linedetector import LineDetector
from darts.detection.circledetector import CircleDetector
from darts.detection.ensembledetector import EnsembleDetector

def run():
    """
    Face and dartboard detection using Viola Jones and Hough Transform methods.
    """
    # thresholding params
    sobel_t_val = 150
    houghlines_t_val = 25
    houghcircles_t_val = 30
    houghcircles_min_r = 50
    houghcircles_max_r = 200
    houghcircles_r_step = 5
    lines_mindist = 10
    circles_mindist = 50
    all_spaces = False

    # load frame from arguments    
    frame_original, name = readfromargs(sys.argv)

    # create greyscale image
    frame = cv.cvtColor(frame_original, cv.COLOR_BGR2GRAY)
    write.write(frame, name + "_gray")

    # viola jones face detection
    print("\nDetecting faces with Viola Jones...")
    facedetector = ViolaJones("frontalface")
    face_boxes = facedetector.find_bounding_boxes(frame_original, name)
    draw.face_boxes(frame_original, face_boxes, name)

    # viola jones dartboard detection
    print("\nDetecting dartboards with Viola Jones...")
    dartboarddetector = ViolaJones("dartboard")  
    dartboard_boxes = dartboarddetector.find_bounding_boxes(frame_original, name)
    draw.dart_boxes(frame_original, dartboard_boxes, name)

    # gaussain blur
    print("\nApplying gaussian blur...")
    gaussian = Gaussian(size=3)
    frame = gaussian.blur(frame)
    write.gaussian(frame, name)

    # sobel edge detectionedges
    print("\nDetecting edges with Sobel edge detector...")
    sobel = Sobel()
    sobel.edgedetection(frame, threshold_val=sobel_t_val)
    write.sobel(sobel, name)

    # hough lines
    print("\nApplying Hough lines transformation...")
    houghlines = HoughLines()
    houghlines.transform(sobel.t_magnitude, sobel.direction)
    houghlines.threshold(threshold_val=houghlines_t_val)
    write.houghlines(houghlines, name)

    # line detection
    print("\nDetecting lines in Hough space...")
    linedetector = LineDetector(houghlines)
    linedetector.detect(min_dist=lines_mindist)
    draw.lines(frame_original, linedetector.lines, name)

    # hough circles
    print("\nApplying Hough circles transformation...")
    houghcircles = HoughCircles(houghcircles_min_r,
                                houghcircles_max_r,
                                houghcircles_r_step)
    houghcircles.transform(sobel.t_magnitude, sobel.direction)
    houghcircles.threshold(threshold_val=houghcircles_t_val)
    houghcircles.sum()
    write.houghcircles(houghcircles, name, all=all_spaces)

    # detect circles
    print("\nDetecting circles in Hough space...")
    circledetector = CircleDetector(houghcircles)
    circledetector.detect(min_dist=circles_mindist)
    draw.circles(frame_original, circledetector.circles, name)

    # TODO ENSEMBLE
    ensembledetector = EnsembleDetector(dartboarddetector,
                                        linedetector,
                                        circledetector)
    ensembledetector.detect(frame)

    # TODO ADDITIONAL METHOD FOR ENSEMBLE

    print("\nComplete!\n")