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


    def detect(self, frame, min_dist=20):

        boards = np.array([])
        
        for box in self.vj_boxes:
            x_min, y_min = box[0], box[1]
            x_max = x_min + box[2]
            y_max = y_min + box[3]

            print(f"\nbox ({x_min}, {y_min}), ({x_max}, {y_max})")

            for circle in self.circles:

                c_r = circle[0]
                c_y = circle[1]
                c_x = circle[2]
                print(f"circle ({c_r}, {c_x}, {c_y})")

                cb_x_min = c_x - c_r
                cb_y_min = c_y - c_r
                cb_x_max = c_x + c_r
                cb_y_max = c_y + c_r
                
                c_box = [cb_x_min, cb_y_min, cb_x_max, cb_y_max]
                print(f"c_box ({cb_x_min}, {cb_y_min}), ({cb_x_max}, {cb_y_max})")

                draw.bb(frame, box, c_box, "testdart0")

                iou = metrics.score_iou(box, c_box)
                print("iou", iou)

                if (c_x > x_min and c_x < x_max and c_y > y_min and c_y < y_max):
                    boards = np.append(boards, box)

            for line in self.lines:
                # print("\nline :", line)

                line_x1 = line[0]
                line_y1 = line[1]
                line_x2 = line[2]
                line_x2 = line[3]

        print("boards", boards)