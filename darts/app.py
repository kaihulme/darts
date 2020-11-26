import cv2 as cv
import darts.io.draw as draw
import darts.io.write as write
from darts.io.read import read
from darts.manipulation.gaussian import Gaussian
from darts.transformation.houghlines import HoughLines
from darts.transformation.houghcircles import HoughCircles
from darts.detection.edgedetector import Sobel
from darts.detection.circledetector import CircleDetector
from darts.detection.violajones import ViolaJones

def run():
    """
    Face and dartboard detection using Viola Jones and Hough Transform methods.
    """
    # name = "coins1"
    # frame_original = read(name, "test", ".png")
    name = "dart14"
    frame_original = read(name, "test")    
    frame = cv.cvtColor(frame_original, cv.COLOR_BGR2GRAY)
    write.write(frame, name + "_gray")

    # viola jones face detection
    # face_clf = ViolaJones("frontalface")
    # for name in test_names:
    # face_boxes = face_clf.find_bounding_boxes(name)
    # face_clf.draw_box(name, face_boxes)

    # gaussain blur
    print("\nApplying gausian blur...")
    gaussian = Gaussian(size=3)
    frame = gaussian.blur(frame)
    write.gaussian(frame, name)

    # sobel edge detection
    print("\nDetecting edges...")
    sobel = Sobel()
    sobel.edgedetection(frame, threshold_val=250)
    write.sobel(sobel, name)

    # hough lines
    print("\nApplying Hough lines transformation...")
    houghlines = HoughLines()
    houghlines.transform(sobel.t_magnitude, sobel.direction)
    houghlines.threshold(threshold_val=20)
    write.houghlines(houghlines, name)

    # TODO LINE DETECTION

    # hough circles
    print("\nApplying Hough circles transformation...")
    houghcircles = HoughCircles(65, 85, 1)
    houghcircles.transform(sobel.t_magnitude)
    houghcircles.threshold(threshold_val=100)
    houghcircles.sum()
    write.houghcircles(houghcircles, name, all=True)

    # TODO CIRCLE DETECTION
    print("\n\nDetecting circles...")
    circledetector = CircleDetector(houghcircles)
    circledetector.detect(min_dist=50)
    draw.circles(frame_original, circledetector.circles, name)

    # TODO ENSEMBLE HOUGH LINES/CIRCLES

    # viola jones dartboard detection
    # # find dartboards    
    # dartboard_clf = ViolaJones("dartboard")
    # for name in test_names:    
        # dartboard_boxes = dartboard_clf.find_bounding_boxes(name)
        # dartboard_clf.draw_box(name, dartboard_boxes) 

    # TODO ENSEMBLE HOUGH / VJ

    # TODO ADDITIONAL METHOD FOR ENSEMBLE

    print("\nComplete!\n")