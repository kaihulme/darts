import cv2 as cv
import numpy as np
import darts.io.write as write

def lines(frame, lines, name):
    """
    Draw circles on frame.
    """
    frame_copy = np.copy(frame)
    for (x1, y1, x2, y2) in lines:
        cv.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    write.lines(frame_copy, name)


def circles(frame, circles, name):
    """
    Draw circles on frame.does the 
    """
    frame_copy = np.copy(frame)
    for (r, y, x) in circles:
        frame_copy = cv.circle(frame_copy, (x, y), r, (0, 255, 0), 2)
    write.circles(frame_copy, name)


def face_boxes(frame, face_boxes, name):
    """
    Draw boxes on frame
    """
    frame_copy = np.copy(frame)
    for (x,y,w,h) in face_boxes:
        frame_copy = cv.rectangle(frame_copy, (x,y), (x+w,y+h), (0,255,0), 2)
    write.face_boxes(frame_copy, name)


def dart_boxes(frame, dart_boxes, name):
    """
    Draw boxes on frame
    """
    frame_copy = np.copy(frame)
    for (x,y,w,h) in dart_boxes:
        frame_copy = cv.rectangle(frame_copy, (x,y), (x+w,y+h), (0,255,0), 2)
    write.dart_boxes(frame_copy, name)


###

def ensemble_boxes(frame, ensemble_boxes, name):
    """
    Draw boxes on frame
    """
    frame_copy = np.copy(frame)
    for (x, y, w, h) in ensemble_boxes:
        # print(box)
        # x, y, w, h = box[0], box[1], box[2], box[3]
        frame_copy = cv.rectangle(frame_copy, (x,y), (x+w,y+h), (0,255,0), 2)
    write.ensemble_boxes(frame_copy, name)