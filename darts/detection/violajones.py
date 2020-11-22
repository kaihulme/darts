import cv2 as cv
import os

class ViolaJones:
    def __init__(self, cascade_type):
        cwd = os.getcwd() # os.path.abspath()
        cascade_dir = cwd + "/darts/resources/cascades/"
        cascade_path = cascade_dir + cascade_type + "/cascade.xml"
        self._cascade_type = cascade_type                
        self._cascade_clf = cv.CascadeClassifier()
        self._cascade_clf.load(cascade_path)
        self._frame_dir = cwd + "/darts/resources/images/test/"
        self._out_dir = cwd + "/darts/out/"

    def find_bounding_boxes(self, name):
        """
        Find boxes in frame using cascafe
        """ 
        frame = load_check_frame(self._frame_dir, name)
        if (frame.any == False) : return False
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray, frame_gray)
        boxes = self._cascade_clf.detectMultiScale(frame_gray)
        return boxes

    def draw_box(self, name, boxes):
        """
        Draw boxes on frame
        """
        frame = load_check_frame(self._frame_dir, name)
        if (frame.any == False) : return False
        for (x,y,w,h) in boxes:
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        path = self._out_dir + name + "_" + self._cascade_type + ".png"
        cv.imwrite(path, frame)


def load_check_frame(dir, name):
    """
    load frame from file name in directory
    """ 
    path = dir + name + ".jpg"
    frame = cv.imread(path)
    if (frame.any == None):
        print ("Error: image", name, "not found!")
        return False
    return frame