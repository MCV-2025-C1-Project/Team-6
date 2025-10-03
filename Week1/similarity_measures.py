import numpy as np
import cv2
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
    return np.sum(((a - b) ** 2) / (a + b))

def histogram_intersection_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the histogram intersection similarity.  
    """
    return np.sum(np.minimum(a,b))

def hellinger_kernel_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Hellinger kernel similarity.
    """
    return np.sum(np.sqrt(a * b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine distance. 
    """
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def bhattacharyya_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the bhattacharyya distance. Used for probability distributions.
    """
    # Necessary to valid probability distributions ?? Like this -> v = v / np.sum(v)

    # Bhattacharrya coefficient
    # bc = np.sum(np.sqrt(a * b))
    
    # return -np.log(bc)
    return cv2.compareHist(a,b, cv2.HISTCMP_BHATTACHARYYA)


def calculate_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    """
    Calculate distance of histogram a and b from a given metric.

    Args:
        a: 1D array histogram
        b: 1D array histogram
        metric: distance function of choice     
    """
    if metric == "euclidean":
        result = euclidean_distance(a, b)
    elif metric == "l1":
        result = l1_distance(a, b)
    elif metric == "chi2":
        result = chi2_distance(a, b)
    elif metric == "histogram_intersection":
        result = histogram_intersection_distance(a, b)
    elif metric == "hellinger":
        result = hellinger_kernel_distance(a, b)
    elif metric == "cosine":
        result = cosine_distance(a, b)
    elif metric == "bhattacharyya":
        result = bhattacharyya_distance(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return result


def compute_similarities(l1: List[np.ndarray], l2: List[np.ndarray], metric: str = "euclidean") -> np.ndarray:
    # TODO: Improve the explanation, returns the similarities.
    similarities = np.empty((len(l1), len(l2)), dtype=object)
    for i, descriptors1 in enumerate(l1):
        for j, descriptors2 in enumerate(l2):
            d = calculate_distance(descriptors1, descriptors2, metric)
            similarities[i, j] = (float(d), j)  # (valor, indice_columna)
    return similarities

    



# if __name__ == "__main__":
#     a = np.array([1,2,3,4], dtype=np.float32)
#     b = np.array([5,6,7,8], dtype=np.float32)

#     print("Euclidean Distance: ", calculate_distance(a,b,"euclidean"))
#     print("L1 Distance: ", calculate_distance(a,b,"l1"))
#     print("Chi2 Distance: ",  calculate_distance(a,b,"chi2"))
#     print("Histogram Intersection Distance: ", calculate_distance(a,b,"histogram_intersection"))
#     print("Hellinger Kernel Distance: ", calculate_distance(a,b,"hellinger"))
#     print("Cosine Distance: ", calculate_distance(a,b,"cosine"))
#     print("Bhattacharyya Distance: ", calculate_distance(a,b,"bhattacharyya"))
