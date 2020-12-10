import os
import cv2 as cv

class ViolaJones:
    def __init__(self, cascade_type):
        cwd = os.getcwd()
        cascade_dir = cwd + "/darts/resources/cascades/"
        cascade_path = cascade_dir + cascade_type + "/cascade.xml"
        self._cascade_type = cascade_type                
        self._cascade_clf = cv.CascadeClassifier()
        self._cascade_clf.load(cascade_path)
        self._frame_dir = cwd + "/darts/resources/images/test/"
        self._out_dir = cwd + "/darts/out/"
        self.boxes = []

    def find_bounding_boxes(self, frame, name):
        """
        Find boxes in frame using cascade
        """ 
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.boxes = self._cascade_clf.detectMultiScale(frame_gray,
                                                        scaleFactor=1.1,
                                                        # minNeighbors=3,
                                                        minSize=(20, 20),
                                                        maxSize=(200, 200))
        return self.boxes
