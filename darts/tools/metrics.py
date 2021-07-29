import numpy as np

def score_iou(a, b):
    """
    Intersection over union = Area(AnB) / Area(AuB)
    """
    # inner box
    inner_x0 = max(a[0], a[0])
    inner_y0 = max(a[1], b[1])
    inner_x1 = min(a[0] + a[2], b[0] + b[2])
    inner_y1 = min(a[1] + a[3], b[1] + b[3])
    # box areas
    a_area = (a[0] + a[2] - a[0] + 1) * (b[1] + b[3] - b[1] + 1)
    b_area = (a[0] + a[2] - b[0] + 1) * (b[1] + b[3] - b[1] + 1)
    inner_area = max(0, inner_x1 - inner_x0 + 1) * max(0, inner_y1 - inner_y0 + 1)
	# intersection over union
    iou = inner_area / (a_area + b_area - inner_area)
    return iou

def avg_iou(true_boxes, pred_boxes):
    total = 0
    for true_box in true_boxes:
        max = 0
        for pred_box in pred_boxes:
            iou = score_iou(true_box, pred_box)
            if (iou > max):
                max = iou
        if max > 0:
            total += max
    if len(true_boxes) == 0:
        return 0
    return total / len(true_boxes)
        

def score_tpr(true_boxes, pred_boxes):
    tps = 0
    fns = 0
    for true_box in true_boxes:
        for pred_box in pred_boxes:
            if notsamebox(true_box, pred_box) and score_iou(true_box, pred_box) > 0.25:
                tps += 1
                break
            fns += 1
    if (tps + fns) == 0:
        return 0
    tpr = tps / (tps + fns)
    return tpr


def score_precision(true_boxes, pred_boxes):
    """
    Precision = TP/(TP+FP)
    """
    tps = get_tps(true_boxes, pred_boxes)
    fps = get_fps(true_boxes, pred_boxes)
    if (tps + fps) == 0:
        return 0
    precision = tps / (tps + fps)
    return precision


def score_recall(true_boxes, pred_boxes):
    """
    Recall = TP/(TP+FN)
    """
    tps = get_tps(true_boxes, pred_boxes)
    fns = get_fns(true_boxes, pred_boxes)
    if (tps + fns) == 0:
        return 0
    recall = tps / (tps + fns)
    return recall


def score_f1(true_boxes, pred_boxes):
    """
    F1 = 2 * (Sensitivity*Precision) / (Sensitivity+Precision)
    """
    precision = score_precision(true_boxes, pred_boxes)
    recall = score_recall(true_boxes, pred_boxes)
    if (recall + precision) == 0:
        return 0
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


def get_tps(true_boxes, pred_boxes):
    tps = 0
    for true_box in true_boxes:
        for pred_box in pred_boxes:
            if notsamebox(true_box, pred_box) and score_iou(true_box, pred_box) > 0.25:
                tps += 1
                break
    return tps


def get_fps(true_boxes, pred_boxes):
    fps = len(pred_boxes)
    for pred_box in pred_boxes:
        for true_box in true_boxes:
            if notsamebox(true_box, pred_box) and score_iou(true_box, pred_box) > 0.25:
                fps -= 1
                break
    return fps


def get_fns(true_boxes, pred_boxes):
    fns = 0
    for true_box in true_boxes:
        p = 0
        for pred_box in pred_boxes:
            if notsamebox(true_box, pred_box) and score_iou(true_box, pred_box) > 0.25:
                break
            p +=1
        if (p == len(pred_boxes)):
            fns += 1
    return fns

def notsamebox(a, b):
    return not np.array_equal(a, b)

    # if (a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]):
        # return True
    # return False

# ignore metrics using true
# negative as hard to define
# for object detection tasks.

# def score_fpr(true_boxes, pred_boxes):
#     fps = 0
#     tns = 0
#     for pred_box in pred_boxes:
#         p = 0
#         for true_box in true_boxes:
#             if (score_iou(true_box, pred_box) > 0.5):
#                 tps += 1
#                 break
#         if (p == 0):
#             fps += 1
#     tpr = fps / (fps + tns)
#     return tpr

# def get_tns(true_boxes, pred_boxes):
    # return tns

# def score_accuracy(a, b):
#     """
#     Accuracy = (TP+TN)/(TP+FP+FN+TN)
#     """
#     tp = score_tp
#     tn = score_tn
#     fp = score_fp
#     fn = score_fn
#     accuracy = (tp + tn) / (tp + fp + fn + tn)
#     return accuracy

# def specificity(a, b):
#     """
#     Specificity = TN/(TN+FP)
#     """
#     return specificity