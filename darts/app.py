import os
import sys
import cv2 as cv
import darts.io.draw as draw
import darts.io.write as write
from darts.io.read import readfromargs
from darts.manipulation.gaussian import Gaussian
from darts.transformation.segment import Segmenter
from darts.transformation.houghlines import HoughLines
from darts.transformation.houghcircles import HoughCircles
from darts.detection.edgedetector import Sobel
from darts.detection.violajones import ViolaJones
from darts.detection.linedetector import LineDetector
from darts.detection.circledetector import CircleDetector
from darts.detection.ensembledetector import EnsembleDetector
from darts.tests.groundtruths import gettruedartboards, gettruefaces
from darts.tests.evaluate import evaluateresults

def run():
    """
    Face and dartboard detection using Viola Jones and Hough Transform methods.
    """

    # clear outputs
    if sys.argv[1] == "clean":
        for name in ["images", "results"]:
            out_dir = os.path.join(os.getcwd(), "darts/out", name)
            for filename in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, filename))
        return

    # parameters for detection #
    gaussian_size = 3 # size of gaussian kernel
    sobel_t_val = 100 # threshold value for sobel edge detection
    houghlines_t_val = 20 # threshold value for hough lines
    houghcircles_t_val = 30 # threshold value for hough circles

    houghcircles_min_r = 20 # minimum hough circle radius
    houghcircles_max_r = 150 # maximum hough circle radius
    houghcircles_r_step = 5 # skip r radii when drawing circles in hough space

    lines_mindist = 20 
    circles_mindist = 100 # minimum distance between circle detections
    ensemble_mindist = 100 # minimum distance between dartboard detections in ensemble method

    all_spaces = True # draw all hough spaces or only sum
    kmeans = False # apply kmeans clustering
    k = 2 # number of kmeans clusters

    ### LOADING and PREPROCESSING STEPS ###

    preprocessing_steps = 2
    if kmeans:
        preprocessing_steps += 1
    print(f"\n[0/{preprocessing_steps}]: PERFORMING PREPROCESSING STEPS")

    # load frame from arguments    
    frame_original, name = readfromargs(sys.argv)
    base_name = name
    if kmeans:
        name = name + "_kmeans"
    frame = frame_original.copy()
    print(f"[1/{preprocessing_steps}]: Loading frame '{name}'...")

    # gaussain blur
    print(f"[2/{preprocessing_steps}]: Applying gaussian blur...")
    gaussian = Gaussian(size=gaussian_size)
    frame = gaussian.blur(frame)
    write.gaussian(frame, name)

    # image segmentation with KMeans
    if kmeans:
        print(f"[3/{preprocessing_steps}]: Applying KMeans clustering...")
        segmenter = Segmenter(k=k)
        frame = segmenter.segment(frame, name)

    # create greyscale image
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    write.write(frame, name + "_gray")

    print("Done.")

    ### DETECTION STEPS ###

    print(f"\n[0/8]: DETECTING")

    # viola jones face detection
    print("[1/8]: Detecting faces with Viola Jones...")
    facedetector = ViolaJones("frontalface")
    vj_face_boxes = facedetector.find_bounding_boxes(frame_original, name)
    draw.face_boxes(frame_original, vj_face_boxes, name)

    # viola jones dartboard detection
    print("[2/8]: Detecting dartboards with Viola Jones...")
    dartboarddetector = ViolaJones("dartboard")  
    vj_dartboard_boxes = dartboarddetector.find_bounding_boxes(frame_original, name)
    draw.dart_boxes(frame_original, vj_dartboard_boxes, name)

    # sobel edge detectionedges
    print("[3/8]: Detecting edges with Sobel edge detector...")
    sobel = Sobel()
    sobel.edgedetection(frame, threshold_val=sobel_t_val)
    write.sobel(sobel, name)

    # hough lines
    print("[4/8]: Applying Hough lines transformation...")
    houghlines = HoughLines()
    houghlines.transform(sobel.t_magnitude, sobel.direction)
    houghlines.threshold(threshold_val=houghlines_t_val)
    write.houghlines(houghlines, name)

    # line detection
    print("[5/8]: Detecting lines in Hough space...")
    linedetector = LineDetector(houghlines)
    linedetector.detect(min_dist=lines_mindist)
    draw.lines(frame_original, linedetector.lines, name)

    # hough circles
    print("[6/8]: Applying Hough circles transformation...")
    houghcircles = HoughCircles(houghcircles_min_r,
                                houghcircles_max_r,
                                houghcircles_r_step)
    houghcircles.transform(sobel.t_magnitude, sobel.direction)
    houghcircles.threshold(threshold_val=houghcircles_t_val)
    houghcircles.sum()
    write.houghcircles(houghcircles, name, all=all_spaces)

    # detect circles
    print("[7/8]: Detecting circles in Hough space...")
    circledetector = CircleDetector(houghcircles)
    circledetector.detect(min_dist=circles_mindist)
    draw.circles(frame_original, circledetector.circles, name)

    # detect dartboards using ensemble of methods
    print("[8/8]: Detecting dartboards in final ensemble...")
    ensembledetector = EnsembleDetector(dartboarddetector,
                                        linedetector,
                                        circledetector)
    ensembledetector.detect(frame, ensemble_mindist)
    draw.ensemble_boxes(frame_original, ensembledetector.boxes, name)

    print("Done.")

    ### EVALUATION STEPS ###

    print("\n[0/3]: EVALUATING.")

    # faces
    print("\n[1/3]: Viola Jones face detection analysis")
    true_faces = gettruefaces(base_name)
    draw.true_face_boxes(frame_original, true_faces, name)
    draw.true_pred_face_boxes(frame_original, true_faces, vj_face_boxes, name)
    vj_faces_results = evaluateresults(true_faces, vj_face_boxes)#, name, "vj_faces")

    # darts
    print("\n[2/3]: Viola Jones dartboard detection analysis")
    true_darts = gettruedartboards(base_name)
    draw.true_dart_boxes(frame_original, true_darts, name)
    draw.true_pred_dart_boxes(frame_original, true_darts, vj_dartboard_boxes, name)
    vj_darts_results = evaluateresults(true_darts, vj_dartboard_boxes)#, name, "vj_darts")

    # draw predictions with ground truths and output metrics
    print("\n[3/3]: Ensemble dartboard detection analysis")
    draw.true_pred_ensemble_boxes(frame_original, true_darts, ensembledetector.boxes, name)
    ensemble_results = evaluateresults(true_darts, ensembledetector.boxes)#, name, "ensemble")

    # write results to csv files
    results = [
        (vj_faces_results, "vj_faces"),
        (vj_darts_results, "vj_darts_results"),
        (ensemble_results, "ensemble")
    ]
    write.evaluation_results(results, name, kmeans)

    print("Done.\n\nComplete!")