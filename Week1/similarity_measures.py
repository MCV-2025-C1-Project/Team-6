import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def l1_distance(a, b):
    return np.sum(np.abs(a - b))

def chi2_distance(a, b):
    return np.sum(((a - b) ** 2) / (a + b))

def histogram_intersection_distance(a, b):
    return np.sum(np.minimum(a,b))

def hellinger_kernel_distance(a, b):
    return np.sum(np.sqrt(a * b))


# if __name__ == "__main__":
#     a = np.array([1,2,3,4], dtype=np.float32)
#     b = np.array([5,6,7,8], dtype=np.float32)

#     print("Euclidean Distance: ", euclidean_distance(a,b))
#     print("L1 Distance: ", l1_distance(a,b))
#     print("Chi2 Distance: ", chi2_distance(a,b))
#     print("Histogram Intersection Distance: ", histogram_intersection_distance(a,b))
#     print("Hellinger Kernel Distance: ", hellinger_kernel_distance(a,b))