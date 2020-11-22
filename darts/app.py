import cv2 as cv
import darts.io.write as write
from darts.io.read import read
from darts.manipulation.gaussian import Gaussian
from darts.transformation.houghcircles import HoughCircles
from darts.detection.edgedetector import Sobel
from darts.detection.violajones import ViolaJones

def run():
    """
    Main application...
    """
    # get grey frame
    name = "dart0"
    frame = read(name, "test")
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # gaussain blur
    print("\nApplying Gausian blur...")
    # gaussian = Gaussian(size=15)
    # frame = gaussian.blur(frame)
    frame = cv.GaussianBlur(frame, (15,15), 0) # faster
    write.gaussian(frame, name)

    # sobel edge detection
    print("\nDetecting edges...")
    sobel = Sobel()
    sobel.edgedetection(frame, threshold_val=60)
    write.sobel(sobel, name)

    # hough circles
    print("\nApplying hough circles transformation")
    houghcircles = HoughCircles(35, 50, 1, 20)
    houghcircles.transform(sobel.t_magnitude)
    houghcircles.sum()
    houghcircles.thresholdspaces(threshold_val=50)
    write.houghcircles(houghcircles, name, True)

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