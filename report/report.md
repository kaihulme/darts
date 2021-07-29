# Dart-Detection

## 1. Viola-Jones object detector

### Face detection

The Viola-Jones object detector works using a cascade of classifiers. After being trained to detect faces it can produce both accurate and fast face detection of images and video. Here are the results of the Viola Jones cascade classifier on some test images. Ground truth bounding boxes are shown in red, with detections by the detector in green.

                                [img1] [img2] [img3] [img4] [img5]

### Performance and measures

Looking at the above results we can see the detector has done very well. It has managed to detect every face without fail. The true positive rate is a measure of the number of correct positive detections over the number of false negatives. Meaning it is a measure of how many faces were detected and is peanalised by missing detections. As it did not miss any faces over these 5 tests the true positive rate is 100%.

However, you will also notice a number of detections for areas in the image not containing faces. Notice how the trophy in the 4th image and the various additional green boxes in the 5th image, these are false positives. False positives are a useful measure as it can show the precise the detector is. A precise detector makes detections you can trust more than one that produces a lot of false positives.

The true positive rate is also known as the recall of the detector. A detector with a high recall rate is able to detect most of the objects it sees, whilst a precise detector is careful with its choices and does not make a lot of misidentifications. There is a trade-off between these two measure and the decision on which to favour depends on the task at hand; a collision detector would likely favour high recall as to avoid as many crashes as possible whilst a doctor would need their diagnosis detection to be as precise as possible as to not misdiagnose a patient. If a model is to be balanced between to two of these measures then the F1 score is a useful indicator of detection performance.

A detector makes a prediction of an object's location in the form of a bounding box, these are what you see in green above. A comparison is made with the ground truth bounding box (in red) using the union over intersection measure (UOC). This measures the amount of overlap between the boxes and can be used to check if a detection is correct. The amount of overlap that gives a positive classification will also depend on the task, but for this I have used 40%. Below you will see the full table of results from the face detector.

Targets     1     0     0     0     1     11    1     1     2     1     0     1     0     1     2     3
Detections  1     0     0     0     1     13    0     1     1     1     2     1     0     1     3     3
TP count    1     0     0     0     1     11    0     1     0     1     0     1     0     1     2     1
FP count    0     0     0     0     0     1     0     0     1     0     2     0     0     0     0     2
FN count    0     0     0     0     0     0     1     0     2     0     0     0     0     0     0     2
Precision   1.00  0.00  0.00  0.00  1.00  0.92  0.00  1.00  0.00  1.00  0.00  1.00  0.00  1.00  0.00  0.33
Recall      1.00  0.00  0.00  0.00  1.00  1.00  0.00  1.00  0.00  1.00  0.00  1.00  0.00  1.00  0.00  0.33
F1 score    1.00  0.00  0.00  0.00  1.00  0.96  0.00  1.00  0.00  1.00  0.00  1.00  0.00  1.00  1.00  0.33
Avg. IOU    0.48  0.00  0.00  0.00  1.00  0.89  0.00  0.72  0.00  0.86  0.00  1.00  0.00  0.99  0.99  0.28

Average precision : 0.45
Average recall    : 0.45
Average F1        : 0.50

High recall alone is not enough indication for a models performance. It is very easy to create a detector which will always detect every object and it can do so with very little logic needed. Imagine such detector which simply detected every possible bounding box in an image as the object of interest. It would of course routinely achieve a high true positive rate, but the large number of false positives would make this detector useless for any task.

## 2. Building a dartboard detector

### Training a cascade

The Viola Jones algorithm is trained in a number of stages. At each stage a new layer will be added to the detector in the form of a cascade. Below you will see a plot showing how training at each of the stages affects the true positive and false positive rates. As you can see the detector starts out detecting all positives. However it clearly favours this over minimising the false positive rate. As you can see after stage 2 the detector has greatly reduced the FPR and has reached almost 0 and so will have a high F1 score by stage 3.

                                        [INSERT training plot]

### Detection performance

Using the now trained dartboard detection cascade on the test images produces the following results:

Targets     1     1     1     1     1     1     1     1     2     1     3     1     1     1     2     1
Detections  1     2     7     3     1     5     1     2     5     4     7     1     1     2     8     1
TP count    1     1     1     1     1     1     1     1     1     1     3     0     1     1     2     1
FP count    1     0     6     2     0     3     1     0     4     3     2     1     0     1     4     0
FN count    0     0     0     0     0     0     0     0     1     0     0     1     0     0     0     0
Precision   1.00  1.00  0.14  0.33  1.00  0.25  1.00  1.00  0.20  0.25  0.60  0.00  1.00  0.50  0.33  1.00
Recall      1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00  0.50  1.00  1.00  0.00  1.00  1.00  1.00  1.00
F1 score    1.00  1.00  0.25  0.50  1.00  0.40  1.00  1.00  0.29  0.40  0.75  0.00  1.00  0.67  0.50  1.00
Avg. IOU    0.75  0.86  0.79  0.65  0.82  0.80  0.89  0.78  0.44  0.96  0.65  0.00  0.85  0.81  0.93  0.93

Average precision : 0.60
Average recall    : 0.90
Average F1        : 0.67

The dartboard detection works remarkably well, especially given the speed at which its performed. We see an overall TPR rate of 90% and a good IOU of . This indicates the method is good at finding dartboards, however for some images it struggles a lot, producing many false positives.

## 3. Integration with shape detection

                                [show gradient image (mag, dir)]
                                [hough transform for 2 best images]
                                [summed circles and some individual]
                                [show circles and lines detected]
                                [show output of dart detection]

Targets     1     1     1     1     1     1     1     1     2     1     3     1     1     1     2     1
Detections  1     1     1     0     1     2     0     1     1     2     4     1     0     1     3     0
TP count    1     1     0     0     1     1     0     1     1     1     2     0     0     1     2     0
FP count    0     0     1     0     0     1     0     1     0     1     1     1     0     0     1     0
FN count    0     0     1     1     0     0     1     0     1     0     1     1     1     0     0     1
Precision   1.00  1.00  0.00  0.00  1.00  0.60  0.00  1.00  1.00  0.50  0.67  0.00  0.00  1.00  0.67  0.00
Recall      1.00  1.00  0.00  0.00  1.00  1.00  0.00  1.00  0.50  1.00  0.67  0.00  0.00  1.00  1.00  0.00
F1 score    1.00  1.00  0.00  0.00  1.00  0.67  0.00  1.00  0.67  0.67  0.67  0.00  0.00  1.00  0.80  0.00
Avg. IOU    0.87  0.88  0.00  0.00  0.88  0.93  0.00  0.94  0.48  0.86  0.53  0.00  0.00  0.88  0.84  0.00

Average precision : 0.62
Average recall    : 0.67
Average F1        : 0.53

The main issues with the detection methods used were down to generalising performance across a range of images. With a few changes to parameters the model can perform near-perfectly on most images, but generalising the methods various inputs results in both decreasing TPR and increasing FPR. I focused a lot on removing false positive and this can be seen across the board, but without the additional detections from the difficult to detect boards they overall average scores have not increased too much. However if you look at individual image results you can see accuracy has been far improved in a lot of ways, just not through average recall and precision due to analysing this metric late in development.

In retrospect I should have looked at average scores from the start rather than focusing on issues with individual images. I think I have a good solution to the problem, but the scores say it was not worthwhile. However my solution does find accurate circles and line locations, not possible before.

### Detection pipeline

                                     [INSERT FLOWCHART]

- Using gaussian smoothing and sobel edge detection along with thresholding I was able to reduce the image to edge points of objects in the image with minimal noise.
- 3 main detection methods were then applied to the image:
  - The Viola-Jones cascade trained on dartboard images was used to highlight areas of focus in my ensemble of methods.
  - A hough transform ,allowing lines to be found. The idea being that points of high line intersection would likely indicate dartboard centres, particularly when found within bounding boxes of the other detectors.
  - A variation on the hough transform used to find circles. This is beneficial as dartboards are generally circular, although this fails when images are taken from an angle.
- These 3 detection methods were then combined in a way so that if 2 of the 3 methods produced bounding boxes in a with an an intersection over union of more than 0.4, a combination of the two boxes would become a detected board.

## 4. Improvements to detection

### Optimisations

The main optimisation I made to my hough transforms was to reduce the angle of theta used to calculate lines and circles whilst incrementing the accumulator. Using the angle of the gradient in the input image obtained from the Sobel edge detection I reduced the amount of noise within the Hough space and so decreased false positive rates due to intersections of points of non-interest accumulating high values being reduced.
  
                                  [HOUGHSPACE] [HOUGHSPACEREDUCED]

### Clustering

As a further extension I decided to try and use KMeans clustering to reduce the colours in the image down to 2. Below you will see the results.

                                          [KMEANS IMAGE]

My thinking was that with less colours in the image it may have helped edge detection and reduce noise which could have helped in the hough transforms. As is with most things, this became a trade-off and improved some areas but negatively impacted others. Although this did not produce the results I had hoped for it did produce interesting results.

### Further improvements

One method that could be beneficial for this task is convolutional neural networks. Although there is not a lot of data available to train with, using pre-trained models with transfer learning may give beneficial results, especially when combined with the methods used in this project. Another machine-learning based approach could analyse the votes from each of the detectors I have used and learn a way to interpret them more accurately than each can their own. This could be implemented through an ensemble method like stacking.
