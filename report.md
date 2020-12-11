# Dart-Detection

Notes on implementation:

- implemented in python
- using cv2, numpy, scipy, sklearn, tqdm

## 1. Viola-Jones object detector

- ground truth and visualisation
  - test face.cpp on images 4, 5, 13, 14, 15
    - draw true red boxes
    - and generated green boxes
- IOU, TFP
  - what is intersection over union
  - calculate IOU for imgs
    - discuss + threshold value
  - what is TPR FPR TNR FNR
  - what is F1 score
    - calculate all for imgs in table
    - + sensitivity & specificity
  - discuss results
    - difficulties assessing TPR meaningfully
    - why it is always possible to get 100% TPR

## 2. Building my own detector

- training performance
  - training tool produces a strong classifier
  - per stage features are added to the classifier
  - explain TPR FPR during training
  - plot TPR and FPR throughout training (for each stage)
  - interpret graph
- testing performance
  - test performance on all test images
  - produce test images
    - detected bounding boxes in green
    - hard-coded bounding boxes in red
    - include 3 in report
  - calculate TPR, FPR, F1, precision, recall for each
    - display in table
    - include averages for each result (look at sklearn averages)
  - interpret results
    - discuss merits and limitations
    - which situations it performs well in
    - why the model may have failed in cases
    - reasons for different TPR

## 3. Integration with shape detectors

- show gradient image (mag, dir)
- hough transform
    - for 2 best images
      - one front on, multi board (14)
      - one side on (which works when adding lines)
    - show hough circles+lines spaces
      - summed circles and some individual
      - show circles and lines detected
- combine with output of viola jones
  - show output of dart detection
    - ground truth in red
    - predicted in green
  - table of scores
    - FPR, F1, precision, recall, sensitivity & specificity
    - the accuracy increase from VJ to the ensemble
  - detection pipeline
    - describe rationale behind method
    - reasoning behind combinations

## 4. Improving my detector
- developing the detector further using additional opencv functionality
  - kmeans colour segmenting pre detection
- explain optimisations for small gradient range
- speed optimisations
- visualise results
  - display an image that is detected well
  - one that is bad with VJ
  - better with VJ+HT
  - and very well done with this additional method
  - show new hough spaces (less noisy)
- evaluate final model performance
  - FPR, F1, precision, recall, sensitivity & specificity
  - improvement percentages over pt2 and pt3.
- what else could have been done
  - CNNs with transfer learning?
