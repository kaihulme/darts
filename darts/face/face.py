import cv2 as cv
import os

def faces(image_name, casade_name):

    # get test image and classifier directories
    dir = os.getcwd()
    image_dir   = dir + "/darts/resources/images/test/"
    cascade_dir = dir + "/darts/resources/cascades/"
    out_dir     = dir + "/darts/out/"

    # get image and cascade paths
    image_path   = image_dir + image_name + ".jpg"
    cascade_path = cascade_dir + casade_name + ".xml"
    out_path     = out_dir + image_name + "/" + casade_name + "/cascade.png"

    # get frame from image
    frame = cv.imread(image_path)
    if (frame.any == None):
        print ("Error: image", image_name, "not found!")
        return False

    # create cascade and load from xml
    cascade = cv.CascadeClassifier()
    cascade.load(cascade_path)

    # detect face in frame with cascade
    faces = detect_faces(frame, cascade)

    # draw faces
    draw_faces(frame, faces, out_path)

    # return faces
    return True

def detect_faces(frame, cascade):

    # greyscale and equalise image
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray, frame_gray)

    # detect faces
    return cascade.detectMultiScale(frame_gray)

def draw_faces(frame, faces, out_dir):

    # for each face draw bounding box
    for (x,y,w,h) in faces:
        print("face :", x, y, w, h)
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    
    # write image with bounding boxes
    cv.imwrite(out_dir, frame)