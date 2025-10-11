"Contains histogram utils to use throughout the project."
import random
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def histogram(data: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    This method computes the histogram of the input data and 
    returns the counts and the indexes of the edges.
    Ignores black pixels (value = 0).
    """
    if data.ndim != 2:
        raise ValueError("data expects an image of shape (H, W).")
    
    H,W = data.shape
    if H < 1 or W < 1:
        raise ValueError("Empty image.")
    
    min_val, max_val = data.min(), data.max()
    if min_val < 0 or max_val > 1:
        raise ValueError(f"Input data should be normalized to [0,1] (given: [{min_val},{max_val}].")
    elif n_bins < 1:
        raise ValueError(f"Number of bins must be positive (given: {n_bins}).")

    # Filter out black pixels (background)
    valid_pixels = data[data > 0]
    if len(valid_pixels) == 0:
        return np.zeros(n_bins, dtype=np.float32)

    # Compute bin edges
    edges = np.linspace(0, 1, n_bins+1) # 1 more edge than bins

    # Get bin index per data value (np.digitize starts at index 1)
    indices = np.digitize(valid_pixels, edges) - 1

    # Ensure indices are within valid range
    indices = np.clip(indices, 0, n_bins-1) 

    # Count ocurrences in each bin
    histogram = np.bincount(indices, minlength=n_bins).astype(np.float32)

    # Return normalized histogram
    return histogram / np.sum(histogram)



def plot_histogram(
    data: np.ndarray,
    n_bins: int = 10,
    title: str = "Histogram",
    outfile: Optional[str] = None) -> None:
    """
    Plot a histogram of the input data.
    """
    # Compute histogram
    hist = histogram(data, n_bins=n_bins)

    # Plot it (save to image if wanted)
    plt.plot(hist)
    plt.xlabel("Bin")
    plt.ylabel("Probability")
    plt.title(title)
    plt.show()

    if outfile:
        plt.savefig(outfile, dpi=200)


def equalize_histogram(probs: np.ndarray = None):
    """
    Implement histogram equalization based on the class' slides.
    """
    if probs is None:
        raise ValueError("equalize_histogram expects a PDF.")
    num_bins = len(probs)
    assert abs(sum(probs) - 1.0) < 1e-6, "Input must be a PDF that sums to 1."
    mapping = np.zeros(num_bins, dtype=np.int64)
    cdf = 0.0 # Cumulative distribution function

    for idx in range(num_bins):
        cdf += probs[idx]
        t = round((num_bins - 1) * cdf)
        if t < 0:
            t = 0
        if t > num_bins - 1: 
            t = num_bins - 1

        mapping[idx] = int(t) # mapping between old and equalized histogram

    out = np.zeros(num_bins, dtype=float)
    for k, p in enumerate(probs):
        out[mapping[k]] += p
    return out, mapping


if __name__=="__main__":
    random.seed(42)

    data = np.array([[1,1,1,1,1,1, 0.4, 0.3 , 0]])
    print(histogram(data))
    plot_histogram(data)

    
    