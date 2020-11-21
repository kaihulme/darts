# import cv2 as cv
import os
import darts.detection.violajones as vj

def run():
    # get test images
    dir = os.getcwd()
    test_dir = dir + "/darts/resources/images/test"    
    test_names = os.listdir(test_dir)
    test_names = [name.split('.')[0] for name in test_names]

    # find faces
    face_clf = vj.ViolaJones("frontalface")
    for name in test_names:
        face_boxes = face_clf.find_bounding_boxes(name)
        face_clf.draw_box(name, face_boxes)
        
    # find dartboards    
    dartboard_clf = vj.ViolaJones("dartboard")
    for name in test_names:    
        dartboard_boxes = dartboard_clf.find_bounding_boxes(name)
        dartboard_clf.draw_box(name, dartboard_boxes)