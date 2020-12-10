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
        self.boxes = np.array([])

    def detect(self, frame, min_dist=20):     
        # for each cascade dartboard box
        for d_box in self.vj_boxes:
            e_box = None
            # for each circle
            for (cr, cy, cx) in self.circles:
                cb_x_min = cx - cr
                cb_y_min = cy - cr
                c_box = (cb_x_min, cb_y_min, 2*cr, 2*cr)
                iou = metrics.score_iou(d_box, c_box)

                print("iou", iou)

                # if circle bounding box IOU with cascade box > 0.5 assume dartboard
                if (iou > 0.4):
                    e_box = np.array(((np.asarray(d_box) + np.asarray(c_box)) / 2).astype('int'))
                    for box in self.boxes:
                        if (boxesdist(e_box, box) > min_dist):
                            self.boxes = np.vstack(((self.boxes, e_box))) if self.boxes.size else np.array([e_box])
                            break
                        else:
                            print("vj and circle box too close")
                else:
                    print("iou too low")
            # if there are enough line intersections with cascade box assume circle
            if (intersectcount(self.lines, d_box) > 4):
                for box in self.boxes:
                    if (boxesdist(e_box, box) > min_dist):
                        self.boxes = np.vstack(((self.boxes, d_box))) if self.boxes.size else np.array([d_box])
                        break
                    else:
                        print("lineinvj too close")
        # for each detected hough circle
        if (len(self.vj_boxes) < len(self.circles)):
            for (cr, cy, cx) in self.circles:
                cb_x_min = cx - cr
                cb_y_min = cy - cr
                c_box = (cb_x_min, cb_y_min, 2*cr, 2*cr)
                # if there are enough line intersections with hough circle assume dartboard
                if (intersectcount(self.lines, c_box) > 4):
                    for box in self.boxes:
                        if (boxesdist(c_box, box) > min_dist):
                            self.boxes = np.vstack(((self.boxes, c_box))) if self.boxes.size else np.array([c_box])
                            break
        else:
            print("same vj as hough circles")


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


def boxesdist(a, b):
    """
    Check distance between two box centres
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_centre = (ax + aw / 2, ay + ah / 2)
    b_centre = (bx + bw / 2, by + bh / 2)
    dist = np.sqrt((ax - bx)**2 + (ay -  by)**2)
    print("dist:", dist)
    return dist
