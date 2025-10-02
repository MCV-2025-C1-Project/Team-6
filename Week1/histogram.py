"This script contains histogram functions to use throughout the project."
import random
import numbers
from typing import Iterable, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def histogram(
    data: Iterable[numbers.Real],
    n_bins: int = 10,
    data_range: Optional[Tuple[float, float]] = None,
    probs: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    This method computes the histogram of the input data and 
    returns the counts, pdf (if selected) and the indexes of the edges.
    Args:
        - data (iterable of numbers) Input data to compute the histogram from.
        - n_bins (int) Number of bins for the histogram to have.
        - data_range (tuple) Range of the histogram
        - probs (bool) Return the PDF (True) or the counts (False).

    Returns:
        - counts (np.ndarray): Number of pixels inside a bin.
        - pdf (np.ndarray): Bin probabilities if probs=True, else None.
        - edges (np.ndarray): The indexes of the edges of each bin.
    """
    data_arr = np.asarray(data)
    if data_arr.size == 0:
        # Treat case where no data was supplied
        if not data_range:
            low, high = 0.0, 1.0
        else:
            low, high = data_range
        if high == low:
            high = low + 1.0
        width = (high - low) / n_bins
        edges = np.array([low + i*width for i in range(n_bins+1)], dtype=float)
        counts = np.zeros(n_bins, dtype=float)
        pdf = np.zeros(n_bins, dtype=float) if probs else None
        return counts, pdf, edges

    if not data_range:
        low = float(np.min(data_arr))
        high = float(np.max(data_arr))
    else:
        low, high = data_range
    
    # Avoid bins with 0 width
    if high == low:
        high = low + 1.0

    width = (high - low) / n_bins
    edges = np.array([low + i*width for i in range(n_bins+1)], dtype=float) # Always 1 more edge than bin

    # Compute the counts
    counts = np.zeros(n_bins, dtype=float)
    total_in_range = 0 
    for x in data_arr:
        if x < low or x > high:
            continue

        idx = int((x - low) / width)
        
        # Fix when idx is out of histogram scope
        if idx == n_bins:
            idx = n_bins - 1
        counts[idx] += 1
        total_in_range += 1

    # Compute PDF
    if probs:
        pdf = np.zeros(n_bins, dtype=float) if total_in_range == 0 else counts / float(total_in_range)
        return counts, pdf, edges
    return counts, None, edges

def plot_histogram(
    data: Iterable[numbers.Real],
    n_bins: int = 10,
    data_range: Optional[Tuple[float, float]] = None,
    probs: bool = False,
    title: str = "Histogram",
    xlabel: str = "Value",
    outfile: Optional[str] = None) -> Optional[str]:
    """
    Plot a histogram of the input data.
    Args:
        - data (iterable of numbers) Input data to plot the histogram from.
        - n_bins (int) Number of bins for the histogram to have.
        - data_range(tuple) Range for the histogram to be.
        - probs (bool) Return the PDF (True) or the counts (False).
        - title (str): Title of the histogram.
        - xlabel (str): Name of the x-axis.
        - outfile (str): The path to save file.

    Returns:
        - outfile (str): The path where file as saved. None if not provided.
    """
    counts, pdf, edges = histogram(data, n_bins=n_bins, data_range=data_range, probs=probs)

    # Draw bars
    widths = np.diff(edges)
    lefts  = edges[:-1]

    fig, ax = plt.subplots()
    ax.bar(lefts, pdf if probs else counts, width=widths, align="edge")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability" if probs else "Count")
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=200)
    plt.show()
    return outfile

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
    # Create test data
    random.seed(42)
    data = [random.gauss(0, 1) for _ in range(70)] + \
            [random.gauss(3, 0.8) for _ in range(30)]

    data = np.array(data)
    
    # Original histogram
    histogram_path = plot_histogram(
        data,
        n_bins=10,
        title="Example Histogram",
        xlabel="Value",
        outfile=None
    )

    # Get PDF and equalization mapping
    counts, pdf, edges = histogram(data, n_bins=10, probs=True) 
    equalized_hist, mapping = equalize_histogram(pdf)

    print("Original:", counts, pdf)
    print("Equalized:", equalized_hist, mapping)

    
    