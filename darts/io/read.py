import os
import cv2 as cv

def read(name, loc, ext=".jpg"):
    path = getpath(name, loc, ext)
    return cv.imread(path)

def getpath(name, loc, ext=".png"):
    """
    Get file path given name and folder name.
    """
    dir = os.getcwd()
    if (loc == "test") : return dir + "/darts/resources/images/test/" + name + ext
    elif (loc == "out") : return dir + "/darts/out/" + name + ext
    else: return False