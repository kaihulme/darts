import cv2 as cv
import numpy as np
import darts.io.write as write

class Segmenter():

    def __init__(self, k=2):
        self.k = k

    def segment(self, frame, name):
        original_frame = frame.copy()
        flat_frame = frame.copy().reshape(-1, 3).astype('float32')
        stopping = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        ret, labels, clusters = cv.kmeans(flat_frame, 
                                          self.k,
                                          None,
                                          stopping,
                                          10,
                                          cv.KMEANS_RANDOM_CENTERS)        
        clusters = np.uint8(clusters)
        clustered_frame = clusters[labels.flatten()]
        clustered_frame = clustered_frame.reshape((frame.shape))
        write.clustered(clustered_frame, name)
        return clustered_frame

        # removedCluster = 1
        # cannyImage = np.copy(original_frame).reshape((-1, 3))
        # cannyImage[labels.flatten() == removedCluster] = [0, 0, 0]
        # cannyImage = cv.Canny(cannyImage,100,200).reshape(original_frame.shape)
        # write.canny(cannyImage, name)
        # ##
        # initialContoursImage = np.copy(cannyImage)
        # imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
        # _, thresh = cv.threshold(imgray, 50, 255, 0)
        # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(initialContoursImage, contours, -1, (0,0,255), cv.CHAIN_APPROX_SIMPLE)
        # write.clustered_canny(initialContoursImage, name)
        # cnt = contours[0]
        # largest_area=0
        # index = 0
        # for contour in contours:
        #     if index > 0:
        #         area = cv.contourArea(contour)
        #         if (area>largest_area):
        #             largest_area=area
        #             cnt = contours[index]
        #     index = index + 1
        # biggestContourImage = np.copy(original_frame)
        # cv.drawContours(biggestContourImage, [cnt], -1, (0,0,255), 3)
        # write.contour(biggestContourImage, name)