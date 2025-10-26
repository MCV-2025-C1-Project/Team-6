import numpy as np
from typing import List

def _to_bool_mask(x: np.ndarray) -> np.ndarray:
    """
    Normalizes an input mask into a 2D boolean (dtype=bool) array.

    - If the mask is 3D (e.g., HxWxC or HxWx3 RGB), it is reduced
      along the last (channel) axis using a logical 'any'. Any pixel
      with a non-zero value in any channel becomes True.
    - If the mask is 2D (HxW), it is simply converted to boolean type.
    - Other dimensions (e.g., 1D or 4D+) will raise a ValueError.
    """
    if x.ndim == 3:
        # Reduce 3D array (HxWxC) to 2D (HxW)
        x = x.any(axis=-1)
    elif x.ndim != 2:
        # Only 2D or 3D inputs are supported
        raise ValueError(f"Unsupported mask shape {x.shape}")
    
    # Cast the final 2D array to boolean (True/False)
    return x.astype(bool)

def precision(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    p = _to_bool_mask(prediction)
    g = _to_bool_mask(ground_truth)
    TP = np.logical_and(p, g).sum()
    FP = np.logical_and(p, np.logical_not(g)).sum()
    return 0.0 if (TP + FP) == 0 else TP / (TP + FP)

def recall(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    p = _to_bool_mask(prediction)
    g = _to_bool_mask(ground_truth)
    TP = np.logical_and(p, g).sum()
    FN = np.logical_and(np.logical_not(p), g).sum()
    return 0.0 if (TP + FN) == 0 else TP / (TP + FN)

def f1_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    prec = precision(prediction, ground_truth)
    rec = recall(prediction, ground_truth)
    return 0.0 if (prec + rec) == 0 else 2 * (prec * rec) / (prec + rec)


# This part of the code can stay the same as Week 2


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


def evaluation(predictions: List[np.ndarray], ground_truths: List[np.ndarray]):

    precision = mean_precision(predictions, ground_truths)
    recall = mean_recall(predictions, ground_truths)
    f1_score = mean_f1_from_metrics(precision, recall)

    return precision, recall, f1_score



if __name__ == "__main__":

    # Create simple binary segmentation arrays (2D)
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

    mean_prec, mean_rec, mean_f1 = evaluation(predictions, ground_truths)

    print("Mean Precision:", mean_prec)
    print("Mean Recall:", mean_rec)
    print("Mean F1 Score:", mean_f1)
