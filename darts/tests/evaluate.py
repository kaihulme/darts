import os
import pandas as pd

import darts.tools.metrics as metrics

def evaluateresults(groundtruths, predictions):
    """
    Evaluate performance of predictions vs. ground truths.
    """
    n_groundtruths = len(groundtruths)
    n_predictions = len(predictions)
    tps, fps, fns, ious = metrics.get_tpfpfn(groundtruths, predictions)
    if n_groundtruths > 0: # if no ground truths then metrics are n/a
        precision = metrics.score_precision(tps, fps)
        recall = metrics.score_recall(tps, fns)
        f1 = metrics.score_f1(precision, recall)
    else:
        precision, recall, f1 = -1, -1, -1
    avg_iou, avg_detect_iou  = metrics.avg_iou(ious, n_groundtruths)

    precision_str = "{}".format(str(round(precision, 2)) if precision >= 0 else 'n/a')
    recall_str = "{}".format(str(round(recall, 2)) if recall >= 0 else 'n/a')
    f1_str = "{}".format(str(round(f1, 2)) if f1 >= 0 else 'n/a')    

    avg_iou_str = "{}".format(str(round(avg_iou, 2)) if avg_iou >= 0 else 'n/a')
    avg_detect_iou_str = "{}".format(str(round(avg_detect_iou, 2)) if avg_iou >= 0 else 'n/a')

    print("+--------------------------+")
    print(f"| Targets           : {n_groundtruths}")
    print(f"| Detections        : {n_predictions}")
    print(f"| TP count          : {tps}")
    print(f"| FP count          : {fps}")
    print(f"| FN count          : {fns}")
    print(f"| Precision         : {precision_str}")
    print(f"| Recall            : {recall_str}")
    print(f"| F1 score          : {f1_str}")
    print(f"| Avg. Detected IOU : {avg_detect_iou_str}")
    print(f"| Avg. Overall IOU  : {avg_iou_str}")
    print("+--------------------------+\n")
    return [
            n_groundtruths, 
            n_predictions, 
            tps, 
            fps, 
            fns, 
            precision, 
            recall, 
            f1, 
            avg_detect_iou,
            avg_iou,
        ]
