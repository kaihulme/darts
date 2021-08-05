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
    precision = metrics.score_precision(tps, fps)
    recall = metrics.score_recall(tps, fns)
    f1 = metrics.score_f1(precision, recall)
    avg_iou, avg_detect_iou  = metrics.avg_iou(ious, n_groundtruths)

    avg_iou_str = "{}".format(str(round(avg_iou, 2)) if avg_iou >= 0 else 'n/a')
    avg_detect_iou_str = "{}".format(str(round(avg_detect_iou, 2)) if avg_iou >= 0 else 'n/a')

    print("+--------------------------+")
    print(f"| Targets           : {n_groundtruths}")
    print(f"| Detections        : {n_predictions}")
    print(f"| TP count          : {tps}")
    print(f"| FP count          : {fps}")
    print(f"| FN count          : {fns}")
    print(f"| Precision         : {precision:.2f}")
    print(f"| Recall            : {recall:.2f}")
    print(f"| F1 score          : {f1:.2f}")
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
