import numpy as np
import darts.io.draw as draw
import darts.tools.metrics as metrics
from darts.io.draw import circles
from darts.tools.utils import localmaxima

class EnsembleDetector():
    def __init__(self, violajones, linedetector, circledetector):
        self.vj_boxes = violajones.boxes
        self.lines = linedetector.lines
        self.circles = circledetector.circles
        self.boxes = None

    def detect(self, frame, min_dist=20):     
        # for each cascade dartboard box
        for d_box in self.vj_boxes:
            # for each circle
            for (cr, cy, cx) in self.circles:
                cb_x_min = cx - cr
                cb_y_min = cy - cr
                c_box = (cb_x_min, cb_y_min, 2*cr, 2*cr)
                iou = metrics.score_iou(d_box, c_box)
                # if circle bounding box IOU with cascade box > 0.5 assume dartboard
                if (iou > 0.4):
                    e_box = np.array(((np.asarray(d_box) + np.asarray(c_box)) / 2).astype('int'))
                    if checknotduplicate(e_box, self.boxes, min_dist):
                        if self.boxes == None:
                            self.boxes = self.boxes.append(e_box)
                        else: 
                            self.boxes = [e_box]
            # if there are enough line intersections with cascade box assume circle
            if (intersectcount(self.lines, d_box) > 4):
                if checknotduplicate(d_box, self.boxes, min_dist):
                    if self.boxes == None:
                        self.boxes = self.boxes.append(d_box)
                    else:
                        self.boxes = [d_box]
        # for each detected hough circle
        for (cr, cy, cx) in self.circles:
            cb_x_min = cx - cr
            cb_y_min = cy - cr
            c_box = (cb_x_min, cb_y_min, 2*cr, 2*cr)
            # if there are enough line intersections with hough circle assume dartboard
            if (intersectcount(self.lines, c_box) > 4):
                if checknotduplicate(c_box, self.boxes, min_dist):
                    if self.boxes == None:
                        self.boxes = self.boxes.append(c_box)
                    else:
                        self.boxes = [c_box]
        n_boards = len(self.boxes)
        if n_boards == 0: print(f"\nNo dartboards detected")
        elif n_boards == 1: print(f"\nDetected a dartboard!")
        else: print(f"\nDetected {n_boards} dartboards!")


def intersectcount(lines, box):
    """
    Get number of intersections within box
    """
    count = 0
    x_min = box[0]
    y_min = box[1]
    x_max = x_min + box[2]
    y_max = y_min + box[3]
    # for each line
    for (x1, y1, x2, y2) in lines:
        # get points of line in box
        coeffs = np.polyfit([x1, x2], [y1, y2], 1)
        linline = np.poly1d(coeffs)
        xs = np.linspace(x_min, x_max, 10)
        ys = linline(xs)
        # increment if line intersects box
        for y in ys:
            if y > y_min and y < y_max:
                count += 1
                break
    return count


def checknotduplicate(a, boxes, min_dist):
    """
    Checks if new box is a duplicate based on distance to other boxes.
    True if less than min_dist to another box already found.
    False if a new box.
    """
    for box in boxes:
        if (boxesdist(a, box) < min_dist):
            return False
    return True


def boxesdist(a, b):
    """
    Check distance between two box centres
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_centre_x, a_centre_y = (ax + aw / 2, ay + ah / 2)
    b_centre_x, b_centre_y = (bx + bw / 2, by + bh / 2)
    dist = np.sqrt((a_centre_x - b_centre_x)**2 +
                   (a_centre_y - b_centre_y)**2)
    return dist
