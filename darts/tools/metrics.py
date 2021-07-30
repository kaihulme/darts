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

def avg_iou(ious):
    if len(ious) == 0:
        return 0
    return np.asarray(ious).mean()

# def avg_iou(true_boxes, pred_boxes):
#     total = 0
#     for true_box in true_boxes:
#         max = 0
#         for pred_box in pred_boxes:
#             iou = score_iou(true_box, pred_box)
#             if (iou > max):
#                 max = iou
#         if max > 0:
#             total += max
#     if len(true_boxes) == 0:
#         return 0
#     return total / len(true_boxes)

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