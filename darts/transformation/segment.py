import cv2 as cv
import numpy as np
import darts.io.write as write

class Segmenter():

    def __init__(self, k=2):
        self.k = k

    def segment(self, frame, name):

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