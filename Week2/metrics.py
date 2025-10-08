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

def precision(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate precision directly from prediction and ground truth arrays
    """
    TP = ((prediction == 1) & (ground_truth == 1)).sum()
    FP = ((prediction == 1) & (ground_truth == 0)).sum()

    # Handle division by zero
    if TP + FP == 0:
        return 0.0
    
    return TP / (TP + FP)

def recall(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate recall directly from prediction and ground truth arrays
    """
    TP = ((prediction == 1) & (ground_truth == 1)).sum()
    FN = ((prediction == 0) & (ground_truth == 1)).sum()

    # Handle division by zero
    if TP + FN == 0:
        return 0.0
    
    return TP / (TP + FN)

def f1_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate F1 score directly from prediction and ground truth arrays
    """
    prec = precision(prediction, ground_truth)
    rec = recall(prediction, ground_truth)
    
    # Handle division by zero
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)

def intersection_over_union(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) directly from prediction and ground truth arrays
    """
    intersection = np.logical_and(prediction == 1, ground_truth == 1).sum()
    union = np.logical_or(prediction == 1, ground_truth == 1).sum()

    # Handle division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

def mean_precision(predictions: List[np.ndarray], ground_truths: List[np.ndarray]) -> float:
    """
    Calculate mean precision across multiple prediction-ground truth pairs
    """
    total_precision = 0.0
    for pred, gt in zip(predictions, ground_truths):
        total_precision += precision(pred, gt)
    
    return total_precision / len(predictions)

def mean_recall(predictions: List[np.ndarray], ground_truths: List[np.ndarray]) -> float:
    """
    Calculate mean recall across multiple prediction-ground truth pairs
    """
    total_recall = 0.0
    for pred, gt in zip(predictions, ground_truths):
        total_recall += recall(pred, gt)
    
    return total_recall / len(predictions)

def mean_f1_score(predictions: List[np.ndarray], ground_truths: List[np.ndarray]) -> float:
    """
    Calculate mean F1 score across multiple prediction-ground truth pairs
    """
    total_f1 = 0.0
    for pred, gt in zip(predictions, ground_truths):
        total_f1 += f1_score(pred, gt)
    
    return total_f1 / len(predictions)

def mean_f1_from_metrics(mean_precision: float, mean_recall: float) -> float:
    """
    Calculate F1 score from pre-computed mean precision and recall
    """
    if mean_precision + mean_recall == 0:
        return 0.0
    return 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)

def mean_iou(predictions: List[np.ndarray], ground_truths: List[np.ndarray]) -> float:
    """
    Calculate mean Intersection over Union (IoU) across multiple prediction-ground truth pairs
    """
    total_iou = 0.0
    for pred, gt in zip(predictions, ground_truths):
        total_iou += intersection_over_union(pred, gt)
    
    return total_iou / len(predictions)

def evaluation(predictions: List[np.ndarray], ground_truths: List[np.ndarray]):

    precision = mean_precision(predictions, ground_truths)
    recall = mean_recall(predictions, ground_truths)
    f1_score = mean_f1_from_metrics(precision, recall)
    iou = mean_iou(predictions, ground_truths)

    return precision, recall, f1_score, iou


if __name__ == "__main__":
    # Create binary segmentation arrays (2D)
    height, width = 10, 10

    predictions = []
    ground_truths = []

    # NORMAL CASE
    ground_truth = np.zeros((height, width), dtype=np.uint8)
    ground_truth[2:6, 2:6] = 1  # Square in the middle
    ground_truth[7:9, 7:9] = 1  # Small square in corner

    # Prediction - slightly different from ground truth
    prediction = np.zeros((height, width), dtype=np.uint8)
    prediction[3:7, 3:7] = 1  # Square shifted slightly
    prediction[6:8, 6:8] = 1  # Small square

    predictions.append(prediction)
    ground_truths.append(ground_truth)

    # PERFECT CASE
    gt2 = np.zeros((height, width), dtype=np.uint8)
    gt2[1:5, 1:5] = 1  # Perfect square
    gt2[8:10, 8:10] = 1  # Corner
    
    pred2 = gt2.copy()  # Perfect prediction
    
    predictions.append(pred2)
    ground_truths.append(gt2)

    # OVER-PREDICTION
    gt4 = np.zeros((height, width), dtype=np.uint8)
    gt4[4:6, 4:6] = 1  # Small square in center
    
    pred4 = np.ones((height, width), dtype=np.uint8)  # Predict everything as positive
    
    predictions.append(pred4)
    ground_truths.append(gt4)

    mean_prec, mean_rec, mean_f1, mean_iou = evaluation(predictions, ground_truths)

    print("Mean Precision:", mean_prec)
    print("Mean Recall:", mean_rec)
    print("Mean F1 Score:", mean_f1)
    print("Mean IoU:", mean_iou)