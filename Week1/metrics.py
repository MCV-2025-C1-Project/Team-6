"""
Implementation of the MAP@k metric explained at the following website: 
https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map

"""
import numpy as np
from typing import List

def precision(query_predictions: np.ndarray, query_gt: List[int]) -> float:
    
    """
    Calculates Precision.
    """

    current_precisions = 0
    for prediction in query_predictions:
        if prediction in query_gt:
            current_precisions += 1

    return current_precisions / len(query_predictions)


def average_precision(query_predictions: np.ndarray, query_gt: List[int]) -> float:

    """
    Calculates Average Precision (AP).
    """

    avg_precision = 0
    relevant_items = 0
    for i, prediction in enumerate(query_predictions):
        if prediction in query_gt:
            relevant_items += 1
            avg_precision += precision(query_predictions[:i + 1], query_gt)
    if relevant_items == 0:
        return 0.0
    return avg_precision / len(query_gt)

# TODO: TRY TO OPTIMIZE THIS COMPUTATION, THOUGH IT IS NOT SUPER IMPORTANT
def mean_average_precision(predictions: np.ndarray, gt: List[List[int]], k=10) -> float:
    """
    Calculates Mean Average Precision at k (MAP@k).

    Args:
        - predictions (np.ndarray): Sorted predictions in shape (n_queries, length_bbdd).
        - gt (List[List[int]]):     Ground truth correspondances per query element.
        - k (int):                  k at which to compute MAP.

    Returns:
        float: MAP@k.
    """
    # Top-k predictions
    top_k_predictions = predictions[:,:k]

    map_score = 0
    for query_predictions, query_gt in zip(top_k_predictions, gt):
        map_score += average_precision(query_predictions, query_gt)

    return map_score / len(top_k_predictions)
