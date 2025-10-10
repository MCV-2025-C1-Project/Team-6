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
        gt_index = np.where(query_predictions == query_gt[0])[0]
        
        # Ground truth predicted: add precision based on rank
        if len(gt_index) > 0:
            map_score += 1 / (gt_index[0] + 1)

    return map_score / len(top_k_predictions)    