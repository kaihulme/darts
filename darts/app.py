# import cv2 as cv
import darts.face.face as face

def run():
    
    # get faces in dart4 using frontalface classifier
    face.faces("dart4", "frontalface")