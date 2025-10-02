"""
Implementation of the MAP@k metric explained at the following website: 
https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map

"""

def precision(predictions, truths):
    
    """
    Calculates Precision.
    """

    current_precisions = 0
    for prediction in predictions:
        if prediction in truths:
            current_precisions += 1

    return current_precisions / len(predictions)


def average_precision(predictions, truths):

    """
    Calculates Average Precision (AP).
    """

    avg_precision = 0
    relevant_items = 0
    for i, prediction in enumerate(predictions):
        if prediction in truths:
            relevant_items += 1
            avg_precision += precision(predictions[:i + 1], truths)
    if relevant_items == 0:
        return 0.0
    return avg_precision / len(truths)

def mean_average_precision(all_predictions, all_truths, k=10):

    """
    Calculates Mean Average Precision at k (MAP@k).
    """

    map_score = 0
    for predictions, truths in zip(all_predictions, all_truths):
        if len(predictions) > k:
            predictions = predictions[:k]
            
        map_score += average_precision(predictions, truths)

    return map_score / len(all_predictions)


