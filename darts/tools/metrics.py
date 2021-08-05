import numpy as np

def overlaps(a_0x, a_0y, a_1x, a_1y, b_0x, b_0y, b_1x, b_1y):
    """
    Check if boxes overlap.
    """
    # no depth or width cannot overlap
    if a_0x == a_1x or a_0y == a_1y or b_0x == b_1x or b_0y == b_1y:
        return False
    # boxes do not overlap horizontally
    if a_1x <= b_0x or b_1x <= a_0x:
        return False
    # boxes do not overlap vertically
    if a_1y <= b_0y or b_1y <= a_0y:
        return False
    # boxes overlap
    return True

def score_iou(a, b):
    """
    Calculate the intersection over union (IOU) metric 
    of two rectangles a and b.

    notation:
    	(0x,0y): (x,y) of top-left corner of rectangle
    	(1x,1y): (x,y) of bottom-right corner of rectangle
    	W: width of rectangle
    	H: height of rectangle
    	A: area of rectangle
    	i: intersection of two rectangles 
    	u: union of two rectangles
    args:
    	Two rectangles a, b of form [0x,0y,W,H] where:
    		(0x,0y) represent coordinates of top-left corner
    		W and H represent width and height respectively
    returns:
    	IOU = A_i / A_u
    """
    # get a_0, b_0 (x,y), width (W) and height (H) 
    a_0x, a_0y, W_a, H_a = a
    b_0x, b_0y, W_b, H_b = b
    # get a_1 (x,y)
    a_1x = a_0x + W_a
    a_1y = a_0y + H_a
    # get b_1 (x,y)
    b_1x = b_0x + W_b
    b_1y = b_0y + H_b
    # if no overlap return IOU = 0
    if not overlaps(a_0x, a_0y, a_1x, a_1y, b_0x, b_0y, b_1x, b_1y):
        return 0
    # get i0 (x,y)
    i_x0 = max(a_0x, b_0x)
    i_y0 = max(a_0y, b_0y)
    # get i1 (x,y)
    i_x1 = min(a_1x, b_1x)
    i_y1 = min(a_1y, b_1y)
    # get i width and height
    W_i = abs(i_x1 - i_x0)
    H_i = abs(i_y1 - i_y0)
    # get a, b and i area (A)
    A_a = W_a * H_a
    A_b = W_b * H_b
    A_i = W_i * H_i
    # get area of union
    A_u = A_a + A_b - A_i
    # calculate intersection over union
    iou = A_i / A_u
    return iou

def avg_iou(ious, n_groundtruths):
    """
    Return average IOU for image.
    """
    if n_groundtruths == 0:
        return -1, -1 # if no ground truths return null marker (-1)
    avg_iou = np.asarray(ious).sum() / n_groundtruths
    if not ious:
        avg_detect_iou = 0.0
    else:
        avg_detect_iou = np.asarray(ious).mean()
    return avg_iou, avg_detect_iou

def get_tpfpfn(true_boxes, pred_boxes):
    """
    Get TPs, FPs and FNs from bounding boxes.
    """
    t_iou = 0.25 # iou threshold for detection
    ious = [] # list of IOUs for TPs
    tps = 0
    true_to_assign = true_boxes.tolist()
    pred_to_assign = pred_boxes.tolist()
    for tb in true_boxes.tolist():
        if tb in true_to_assign:
            max = []
            max_iou = 0
            for pb in pred_boxes.tolist():
                if pb in pred_to_assign:
                    iou = score_iou(tb, pb)
                    if iou > max_iou:
                        max = pb
                        max_iou = iou
            if max_iou > t_iou:
                tps += 1
                true_to_assign.remove(tb)
                pred_to_assign.remove(max)
                ious.append(max_iou)
    fns = len(true_to_assign)
    fps = len(pred_to_assign)
    return tps, fps, fns, ious


def score_precision(tps, fps):
    """
    Precision = TP/(TP+FP)
    """
    if (tps + fps) == 0:
        return 0
    return tps / (tps + fps)


def score_recall(tps, fns):
    """
    Recall = TP/(TP+FN)
    """
    if (tps + fns) == 0:
        return 0
    return tps / (tps + fns)


def score_f1(precision, recall):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    if (recall + precision) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)