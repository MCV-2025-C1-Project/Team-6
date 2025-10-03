import numpy as np
from typing import List


def mean_average_precision(predictions: np.ndarray, gt: List[List[int]], k=10) -> float:
    """
    Compute mAP@K score for the given predictions and ground truth.
    This functions takes into account that there is only one relevant item per query.

    Args:
        predictions (np.ndarray): Array of shape (num_queries, num_items) with predicted item indices sorted by relevance.
        gt (List[List[int]]): List of lists containing the ground truth relevant item index for each query.
        k (int): The number of top predictions to consider for each query.
    
    Returns:
        float: The mean Average Precision at K score.
    """

    top_k_predictions = predictions[:, :k]

    map_score = 0
    for query_predictions, query_gt in zip(top_k_predictions, gt):
        relevant = query_gt[0]
        
        if relevant in query_predictions:
            index = np.where(query_predictions == relevant)
            map_score += 1 / (index[0][0] + 1)
        else:
            map_score += 0

    return map_score / len(top_k_predictions)    