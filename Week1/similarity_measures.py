import numpy as np
from typing import List


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Euclidean (L2) distance between two vectors.
    """
    return np.sqrt(np.sum((a - b) ** 2))

def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Manhattan (L1) distance between two vectors.
    """
    return np.sum(np.abs(a - b))

def chi2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Chi-square distance between two vectors.
    """
    return np.sum(((a - b) ** 2) / (a + b + 1e-10))

def histogram_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the histogram intersection similarity. Used for probability distributions.
    """
    return -np.sum(np.minimum(a,b)) # Higher the better, negate it

def hellinger_kernel(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Hellinger kernel similarity. Used for probability distributions.
    """
    return -np.sum(np.sqrt(a * b)) # Higher the better, negate it

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine distance. 
    """
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def bhattacharyya_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the bhattacharyya distance. Used for probability distributions.
    """
    bc = np.sum(np.sqrt(a * b)) # Bhattacharrya coefficient
    return -np.log(bc + 1e-10)

def compute_similarity(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    """
    Calculate similarity for descriptors a and b from a given metric. 
    The lower the metric, the better.

    Args:
        a (np.ndarray): 1D array histogram
        b (np.ndarray): 1D array histogram
        metric (str):   metric of choice

    Returns:
        float: Metric value.
    """
    if metric == "euclidean":
        result = euclidean_distance(a, b)
    elif metric == "l1":
        result = l1_distance(a, b)
    elif metric == "chi2":
        result = chi2_distance(a, b)
    elif metric == "histogram_intersection":
        result = histogram_intersection(a, b)
    elif metric == "hellinger":
        result = hellinger_kernel(a, b)
    elif metric == "cosine":
        result = cosine_distance(a, b)
    elif metric == "bhattacharyya":
        result = bhattacharyya_distance(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return result

def compute_similarities(l1: List[np.ndarray], l2: List[np.ndarray], metric: str = "euclidean") -> np.ndarray:
    """
    Compute similarities between image descriptors of 2 lists (with respect to a metric).

    Args:
        l1 (List[np.ndarray]): First list of image descriptors.
        l2 (List[np.ndarray]): Second list of image descriptors.
        metric (str):          metric of choice  

    Returns:   
        np.ndarray: Array of similarities of shape (len(l1), len(l2))
    """
    similarities = np.empty((len(l1), len(l2)), dtype=object)
    for i, descriptors1 in enumerate(l1):
        for j, descriptors2 in enumerate(l2):
            similarities[i, j] = compute_similarity(descriptors1, descriptors2, metric)
    return similarities
