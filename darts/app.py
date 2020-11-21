# import cv2 as cv
import os
import cv2 as cv
import darts.detection.violajones as vj

from darts.manipulation.gaussian import Gaussian

def run():

    dir = os.getcwd()
    frame_path = dir + "/darts/resources/images/test/dart0.jpg"
    out_path = dir + "/darts/out/testout.png"

    frame = cv.imread(frame_path)

    gaussian = Gaussian(3)
    frame = gaussian.blur(frame)

    cv.imwrite(out_path, frame)

    # # get test images
    # dir = os.getcwd()
    # test_dir = dir + "/darts/resources/images/test"    
    # test_names = os.listdir(test_dir)
    # test_names = [name.split('.')[0] for name in test_names]
# 
    # # find faces
    # face_clf = vj.ViolaJones("frontalface")
    # for name in test_names:
        # face_boxes = face_clf.find_bounding_boxes(name)
        # face_clf.draw_box(name, face_boxes)
        # 
    # # find dartboards    
    # dartboard_clf = vj.ViolaJones("dartboard")
    # for name in test_names:    
        # dartboard_boxes = dartboard_clf.find_bounding_boxes(name)
        # dartboard_clf.draw_box(name, dartboard_boxes)