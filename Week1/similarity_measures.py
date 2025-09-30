import numpy as np

def euclidean_distance(a, b):
	"""
	Compute the Euclidean (L2) distance between two vectors.
	"""
    return np.sqrt(np.sum((a - b) ** 2))

def l1_distance(a, b):
    """
	Compute the Manhattan (L1) distance between two vectors.
	"""
	return np.sum(np.abs(a - b))

def chi2_distance(a, b):
    """
	Compute the Chi-square distance between two vectors.
	"""
	return np.sum(((a - b) ** 2) / (a + b))

def histogram_intersection_distance(a, b):
    """
	Compute the histogram intersection similarity.
	"""
	return np.sum(np.minimum(a,b))

def hellinger_kernel_distance(a, b):
    """
	Compute the Hellinger kernel similarity.
	"""
	return np.sum(np.sqrt(a * b))

def calculate_distance(a, b, metric:str) -> float:
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
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return result


# if __name__ == "__main__":
#     a = np.array([1,2,3,4], dtype=np.float32)
#     b = np.array([5,6,7,8], dtype=np.float32)

#     print("Euclidean Distance: ", calculate_distance(a,b,"euclidean"))
#     print("L1 Distance: ", calculate_distance(a,b,"l1"))
#     print("Chi2 Distance: ",  calculate_distance(a,b,"chi2"))
#     print("Histogram Intersection Distance: ", calculate_distance(a,b,"histogram_intersection"))
#     print("Hellinger Kernel Distance: ", calculate_distance(a,b,"hellinger"))
