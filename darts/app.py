import cv2 as cv
from darts.manipulation.gaussian import Gaussian
from darts.detection.edgedetector import Sobel
from darts.transformation.houghcircles import HoughCircles
from darts.detection.violajones import ViolaJones
from darts.manipulation.utils import normalisewrite, getpath, radtodeg

def run():
    """
    Main application...
    """
    # get grey frame
    test_name = "coins1"
    frame = cv.imread(getpath(test_name, "test"))
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # gaussain blur
    print("\nApplying Gausian blur...")
    # gaussian = Gaussian(size=15)
    # frame = gaussian.blur(frame)
    frame = cv.GaussianBlur(frame, (15,15), 0) # faster
    normalisewrite(frame, test_name+"_gaussian")

    # sobel edge detection
    print("\nDetecting edges...")
    sobel = Sobel()
    sobel.edgedetection(frame, threshold_val=60)
    normalisewrite(sobel.dfdx, test_name+"_dfdx")
    normalisewrite(sobel.dfdy, test_name+"_dfdy")
    normalisewrite(sobel.magnitude, test_name+"_magnitude")
    normalisewrite(radtodeg(sobel.direction), test_name+"_direction")
    normalisewrite(sobel.t_magnitude, test_name+"_threshold_magnitude")

    # hough circles
    print("\nApplying hough circles transformation")
    houghcircles = HoughCircles(30, 50, 5, 20)
    houghcircles.transform(sobel.t_magnitude)

    print(houghcircles.hough_space.shape)

    s = 0
    for space in houghcircles.hough_space:
        normalisewrite(space, test_name+"_houghspace_" + str(s))
        s += 1

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