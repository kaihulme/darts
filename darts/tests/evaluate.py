import darts.tools.metrics as metrics

def evaluateresults(groundtruths, predictions):
    """
    Evaluate performance of predictions vs. ground truths.
    """
    n_groundtruths = len(groundtruths)
    n_predictions = len(predictions)
    print(f"\nNumber of real targets : {n_groundtruths}")
    print(f"Number of predictions  :  {n_predictions}")
    tps = metrics.get_tps(groundtruths, predictions)
    fps = metrics.get_fps(groundtruths, predictions)
    fns = metrics.get_fns(groundtruths, predictions)
    print(f"\nTrue positive count  : {tps}")
    print(f"False positive count : {fps}")
    print(f"False negative count : {fns}")
    tpr = metrics.score_tpr(groundtruths, predictions)
    precision = metrics.score_precision(groundtruths, predictions)
    recall = metrics.score_recall(groundtruths, predictions)
    f1 = metrics.score_f1(groundtruths, predictions)
    print(f"\nTrue positive rate : {tpr}")
    print(f"Precision          : {precision}")
    print(f"Recall             : {recall}")
    print(f"F1 score           : {f1}")