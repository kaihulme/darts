from darts.io.draw import circles
import numpy as np
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

                print(f"circle ({c_x}, {c_y})")

                print(c_x > x_min)
                print(c_x < x_max)
                print(c_y > y_min)
                print(c_y < y_max)

                if (c_x > x_min and c_x < x_max and c_y > y_min and c_y < y_max):
                    boards = np.append(boards, box)
                    print("in box")
                else:   
                    print("not in box")

            for line in self.lines:
                # print("\nline :", line)

                line_x1 = line[0]
                line_y1 = line[1]
                line_x2 = line[2]
                line_x2 = line[3]

        print("boards", boards)