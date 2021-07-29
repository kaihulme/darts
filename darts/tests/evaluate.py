import os
import pandas as pd

import darts.tools.metrics as metrics

def evaluateresults(groundtruths, predictions): #, name, test):
    """
    Evaluate performance of predictions vs. ground truths.
    """
    n_groundtruths = len(groundtruths)
    n_predictions = len(predictions)
    print("---------------------")
    print(f"| Targets         : {n_groundtruths}    |")
    print(f"| Detections      : {n_predictions}    |")
    tps = metrics.get_tps(groundtruths, predictions)
    fps = metrics.get_fps(groundtruths, predictions)
    fns = metrics.get_fns(groundtruths, predictions)
    print(f"| TP count        : {tps}    |")
    print(f"| FP count        : {fps}    |")
    print(f"| FN count        : {fns}    |")
    tpr = metrics.score_tpr(groundtruths, predictions)
    precision = metrics.score_precision(groundtruths, predictions)
    recall = metrics.score_recall(groundtruths, predictions)
    f1 = metrics.score_f1(groundtruths, predictions)
    avg_iou = metrics.avg_iou(groundtruths, predictions)
    # print(f"| TPR        : {tpr}   |")
    print(f"| Precision (TPR) : {precision:.2f} |")
    print(f"| Recall          : {recall:.2f} |")
    print(f"| F1 score        : {f1:.2f} |")
    print(f"| Avg. IOU        : {avg_iou:.2f} |")
    print("---------------------")

    # write to csv
    # results_df = pd.DataFrame(
        # [[
        #     n_groundtruths, 
        #     n_predictions, 
        #     tps, 
        #     fps, 
        #     fns, 
        #     precision, 
        #     recall, 
        #     f1, 
        #     avg_iou,
        # ]], columns=[
        #     "targets",
        #     "detections",
        #     "tp_count",
        #     "fp_count",
        #     "fn_count",
        #     "precision",
        #     "recall",
        #     "f1_score",
        #     "avg_iou",
        # ]
    # )

    # out_name = "/darts/out/results/{}_{}_results.csv".format(name, test)
    # out_path = os.getcwd() + out_name

    # print("\n", out_name)
    # print(out_path, "\n")

    # results_df.to_csv(out_path)

    return [
            n_groundtruths, 
            n_predictions, 
            tps, 
            fps, 
            fns, 
            precision, 
            recall, 
            f1, 
            avg_iou,
        ]#, [
        #     "targets",
        #     "detections",
        #     "tp_count",
        #     "fp_count",
        #     "fn_count",
        #     "precision",
        #     "recall",
        #     "f1_score",
        #     "avg_iou",
        # ]
