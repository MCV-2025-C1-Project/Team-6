"This script contains utils used throughout the project."
import random
import numbers
from typing import Iterable, List, Tuple, Optional

import matplotlib.pyplot as plt


def histogram(
    data: Iterable[numbers.Real],
    n_bins: int = 10,
    data_range: Optional[Tuple[float, float]] = None,
    probs: bool = False) -> Tuple[List[float], List[float]]:
    """
    This method computes the histogram of the input data and 
    returns the counts and the indexes of the edges.
    Args:
        - data (iterable of numbers) Input data to compute the histogram from.
        - n_bins (int) Number of bins for the histogram to have.
        - data_range (tuple) Range of the histogram
        - probs (bool) Return the PDF (True) or the counts (False).

    Returns:
        - counts (list): Number of pixels inside a bin.
        - pdf (list): Bin probabilities if probs=True, else None.
        - edges (list): The indexes of the edges of each bin.
    """
    if not data:
        return [0]*n_bins, [0]*(n_bins+1)

    data = list(data)

    if not data_range:
        low = min(data)
        high = max(data)
    else:
        low, high = data_range
    
    # avoid 0 width
    if high == low:
        high = low + 1.0

    width = (high - low) / n_bins
    edges = [low + i*width for i in range(n_bins+1)] # always 1 more edge than bin

    # Compute the counts
    counts = [0]*n_bins
    for x in data:
        if x < low or x > high:
            continue

        idx = int((x - low) / width)
        
        # Fix when idx is out of histogram scope
        if idx == n_bins:
            idx = n_bins - 1
        counts[idx] += 1

    # Compute PDF
    if probs:
        n = len(data)
        pdf = [c / n for c in counts]
        return counts, pdf, edges
    
    return counts, None, edges

def plot_histogram(
    data: Iterable[numbers.Real],
    n_bins: int = 10,
    data_range: Optional[Tuple[float, float]] = None,
    probs: bool = False,
    title: str = "Histogram",
    xlabel: str = "Value",
    outfile: Optional[str] = None):
    """
    Plot a histogram of the input data.
    Args:
        - data (iterable of numbers) Input data to plot the histogram from.
        - n_bins (int) Number of bins for the histogram to have.
        - probs (bool) Return the PDF (True) or the counts (False).
        - title (str): Title of the histogram.
        - xlabel (str): Name of the x-axis.
        - outfile (str): The path to save file.

    Returns:
        - outfile (str): The path to save file.
    """
    counts, pdf, edges = histogram(data, n_bins=n_bins, data_range=data_range, probs=probs)

    # Draw bars
    widths = [edges[i+1] - edges[i] for i in range(len(edges)-1)]
    lefts  = edges[:-1]

    fig, ax = plt.subplots()
    ax.bar(lefts, pdf if probs else counts, width=widths, align="edge")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density" if probs else "Count")
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=200)
    plt.show()
    return outfile

def equalize_histogram(probs: list):
    """
    Implement histogram equalization based on the class' slides.
    """
    num_bins = len(probs)
    assert abs(sum(probs) - 1.0) < 1e-6, "Input must be a PDF that sums to 1."
    mapping = [0]*(num_bins)
    cdf = 0.0

    for idx in range(num_bins):
        cdf += probs[idx]
        t = round((num_bins - 1) * cdf)
        if t < 0: t = 0
        if t > num_bins - 1: 
            t = num_bins - 1

        mapping[idx] = int(t) # mapping between old and new hist

    out = [0.0] * len(probs)
    for k, p in enumerate(probs):
        out[mapping[k]] += p
    return out, mapping


if __name__=="__main__":
    # Example

    # Create test data
    random.seed(42)
    data = [random.gauss(0, 1) for _ in range(70)] + \
            [random.gauss(3, 0.8) for _ in range(30)]

    # Plot histogram
    path = plot_histogram(
        data,
        n_bins=10,
        probs=False,
        data_range = (0,1),
        title="Example Histogram",
        xlabel="Value",
        outfile=None
    )

    counts, edges = histogram(data)

    print(counts, edges)