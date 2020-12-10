import os
import sys
import cv2 as cv
import darts.io.draw as draw
import darts.io.write as write
from darts.io.read import readfromargs
from darts.manipulation.gaussian import Gaussian
from darts.transformation.segment import Segmenter
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
    # parameters for detection
    gaussian_size = 3
    sobel_t_val = 100
    houghlines_t_val = 30
    houghcircles_t_val = 30
    houghcircles_min_r = 50
    houghcircles_max_r = 200
    houghcircles_r_step = 1
    lines_mindist = 20
    circles_mindist = 100
    ensemble_mindist = 100
    all_spaces = False
    kmeans = False
    k = 2

    # load frame from arguments    
    frame_original, name = readfromargs(sys.argv)
    frame = frame_original.copy()
    print(f"\nDetecting dartboards in {name}...")

    # gaussain blur
    print("\n[1/9]: Applying gaussian blur...")
    gaussian = Gaussian(size=gaussian_size)
    frame = gaussian.blur(frame)
    write.gaussian(frame, name)

    # image segmentation with KMeans
    if kmeans:
        segmenter = Segmenter(k=k)
        frame = segmenter.segment(frame, name)

    # create greyscale image
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    write.write(frame, name + "_gray")

    # viola jones face detection
    print("[2/9]: Detecting faces with Viola Jones...")
    facedetector = ViolaJones("frontalface")
    face_boxes = facedetector.find_bounding_boxes(frame_original, name)
    draw.face_boxes(frame_original, face_boxes, name)

    # viola jones dartboard detection
    print("[3/9]: Detecting dartboards with Viola Jones...")
    dartboarddetector = ViolaJones("dartboard")  
    dartboard_boxes = dartboarddetector.find_bounding_boxes(frame_original, name)
    draw.dart_boxes(frame_original, dartboard_boxes, name)

    # sobel edge detectionedges
    print("[4/9]: Detecting edges with Sobel edge detector...")
    sobel = Sobel()
    sobel.edgedetection(frame, threshold_val=sobel_t_val)
    write.sobel(sobel, name)

    # hough lines
    print("[5/9]: Applying Hough lines transformation...")
    houghlines = HoughLines()
    houghlines.transform(sobel.t_magnitude, sobel.direction)
    houghlines.threshold(threshold_val=houghlines_t_val)
    write.houghlines(houghlines, name)

    # line detection
    print("[6/9]: Detecting lines in Hough space...")
    linedetector = LineDetector(houghlines)
    linedetector.detect(min_dist=lines_mindist)
    draw.lines(frame_original, linedetector.lines, name)

    # hough circles
    print("[7/9]: Applying Hough circles transformation...")
    houghcircles = HoughCircles(houghcircles_min_r,
                                houghcircles_max_r,
                                houghcircles_r_step)
    houghcircles.transform(sobel.t_magnitude, sobel.direction)
    houghcircles.threshold(threshold_val=houghcircles_t_val)
    houghcircles.sum()
    write.houghcircles(houghcircles, name, all=all_spaces)

    # detect circles
    print("[8/9]: Detecting circles in Hough space...")
    circledetector = CircleDetector(houghcircles)
    circledetector.detect(min_dist=circles_mindist)
    draw.circles(frame_original, circledetector.circles, name)

    # detect dartboards using ensemble of methods
    print("[9/9]: Detecting dartboards in final ensemble...")
    ensembledetector = EnsembleDetector(dartboarddetector,
                                        linedetector,
                                        circledetector)
    ensembledetector.detect(frame, ensemble_mindist)
    draw.ensemble_boxes(frame_original, ensembledetector.boxes, name)

    # TODO BlobDetector (ellipse)

    # TODO Print metrics!

    print("\nComplete!")