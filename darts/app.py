import cv2 as cv
from darts.manipulation.gaussian import Gaussian
from darts.detection.edgedetector import Sobel
from darts.detection.violajones import ViolaJones
from darts.manipulation.utils import normalisewrite, getpath

def run():
    # get grey frame
    test_name = "coins1"
    frame = cv.imread(getpath(test_name, "test"))
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # gaussain blur
    gaussian = Gaussian(5)
    frame = gaussian.blur(frame)

    normalisewrite(frame, test_name+" gaussian")
    # cv.imwrite(getpath("gaussian", "out"), frame)

    # sobel edge detection
    sobel = Sobel()
    sobel.edgedetection(frame)

    normalisewrite(sobel.dfdx, test_name+" dfdx")
    normalisewrite(sobel.dfdy, test_name+" dfdy")
    # cv.imwrite(getpath(test_name+" dfdx", "out"), sobel.dfdx)
    # cv.imwrite(getpath(test_name+" dfdy", "out"), sobel.dfdy)

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