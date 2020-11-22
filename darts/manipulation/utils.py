import os
import cv2 as cv

def normalise(frame):
    """
    minmax normalisation of frame between 0-255
    """
    min, max, _, _ = cv.minMaxLoc(frame)
    rows, cols = frame.shape
    for y in range(rows-1):
        for x in range(cols-1):
            val = frame[y][x]
            # print(f"255 * ({val}-{min})/({max}-{min})")
            if (max-min > 0) : frame[y][x] = 255 * (val-min)/(max-min)
            else : frame[y][x] = 0

    return frame

def normalisewrite(frame, name):

    # cv.imwrite(os.getcwd()+"/darts/out/b4norm_"+name+".png", frame)

    frame = normalise(frame)
    path = getpath(name, "out")
    cv.imwrite(path, frame)

def getpath(name, loc):
    dir = os.getcwd()
    if (loc == "test") : return dir + "/darts/resources/images/test/" + name + ".png"
    elif (loc == "out") : return dir + "/darts/out/" + name + ".png"
    else: return False