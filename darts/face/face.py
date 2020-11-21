import cv2 as cv
import os

def faces(image_name, casade_name):

    # get test image and classifier directories
    dir = os.getcwd()
    test_img_dir = dir + "/resources/images/test/"
    opencv_dir   = dir + "/resources/opencv/"
    out_dir      = dir + "/out/"

    # get image and cascade paths
    image_path   = test_img_dir + image_name + ".jpg"
    cascade_path = opencv_dir + casade_name + ".xml"
    out_path     = out_dir + image_name + "_faces.png"

    print(out_path)

    # get frame from image
    frame = cv.imread(image_path)
    if (frame.all == None):
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

# get faces in dart4 using frontalface classifier
faces("dart4", "frontalface")