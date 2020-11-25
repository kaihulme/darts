import cv2 as cv
import darts.io.write as write
from darts.io.read import read
from darts.manipulation.gaussian import Gaussian
from darts.transformation.houghlines import HoughLines
from darts.transformation.houghcircles import HoughCircles
from darts.detection.edgedetector import Sobel
from darts.detection.violajones import ViolaJones

def run():
    """
    Main application...
    """
    name = "coins1"
    frame = read(name, "test", ".png")
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    write.write(frame, name + "_gray")

    # gaussain blur
    print("\nApplying gausian blur...")
    gaussian = Gaussian(size=15)
    frame = gaussian.blur(frame)
    write.gaussian(frame, name)

    # sobel edge detection
    print("\nDetecting edges...")
    sobel = Sobel()
    sobel.edgedetection(frame, threshold_val=60)
    write.sobel(sobel, name)

    # hough lines
    print("\nApplying Hough lines transformation...")
    houghlines = HoughLines()
    houghlines.transform(sobel.t_magnitude, sobel.direction)
    houghlines.threshold(threshold_val=10)
    write.houghlines(houghlines, name)

    # hough circles
    print("\nApplying Hough circles transformation...")
    houghcircles = HoughCircles(35, 50, 1, 20)
    houghcircles.transform(sobel.t_magnitude)
    houghcircles.sum()
    houghcircles.threshold(threshold_val=70)
    write.houghcircles(houghcircles, name, all=False)

    # # get test images
    # dir = os.getcwd()
    # test_dir = dir + "/darts/resources/images/test"    
    # test_names = os.listdir(test_dir)
    # test_names = [name.split('.')[0] for name in test_names]

    # # find faces
    # face_clf = ViolaJones("frontalface")
    # for name in test_names:
        # face_boxes = face_clf.find_bounding_boxes(name)
        # face_clf.draw_box(name, face_boxes)
    # # find dartboards    
    # dartboard_clf = ViolaJones("dartboard")
    # for name in test_names:    
        # dartboard_boxes = dartboard_clf.find_bounding_boxes(name)
        # dartboard_clf.draw_box(name, dartboard_boxes) 

    print("\nComplete!\n")