import numpy as np
import darts.tools.metrics as metrics

class EnsembleDetector():
    def __init__(self, violajones, linedetector, circledetector):
        self.vj_boxes = violajones.boxes
        self.lines = linedetector.lines
        self.circles = circledetector.circles
        self.boxes = np.array([])

    def detect(self, frame, min_dist=20):     
        # for each detected hough circle
        for (cr, cy, cx) in self.circles:
            cb_x_min = cx - cr
            cb_y_min = cy - cr
            c_box = np.array([cb_x_min, cb_y_min, 2*cr, 2*cr])
            # if there are enough line intersections with hough circle assume dartboard
            if (intersectcount(self.lines, c_box) > 4):
                if checknotduplicate(c_box, self.boxes, min_dist):
                    if (self.boxes.size == 0):
                        self.boxes =  np.asarray([c_box])
                    else:
                        self.boxes = np.vstack((self.boxes, c_box))
        # for each cascade dartboard box
        for d_box in self.vj_boxes:
            # for each circle
            for (cr, cy, cx) in self.circles:
                cb_x_min = cx - cr
                cb_y_min = cy - cr
                c_box = np.array([cb_x_min, cb_y_min, 2*cr, 2*cr])
                iou = metrics.score_iou(d_box, c_box)
                # if circle bounding box IOU with cascade box > 0.5 assume dartboard
                if (iou > 0.4):
                    # e_box = np.array(((np.asarray(d_box) + np.asarray(c_box)) / 2).astype('int'))
                    e0 = int((d_box[0] + c_box[0]) / 2)
                    e1 = int((d_box[1] + c_box[1]) / 2)
                    e2 = int((d_box[2] + c_box[2]) / 2)
                    e3 = int((d_box[3] + c_box[3]) / 2)
                    e_box = np.array([e0, e1, e2, e3])
                    if checknotduplicate(e_box, self.boxes, min_dist):
                        if (self.boxes.size == 0):
                            self.boxes = np.asarray([e_box])
                        else:
                            self.boxes = np.vstack((self.boxes, e_box))
            # if there are enough line intersections with cascade box assume circle
            if (intersectcount(self.lines, d_box) == 0):
                if checknotduplicate(d_box, self.boxes, min_dist):
                    if (self.boxes.size == 0):
                        self.boxes = np.asarray([d_box])
                    else:
                        self.boxes = np.vstack((self.boxes, d_box))


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
    if (boxes.size == 0):
        return True
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
