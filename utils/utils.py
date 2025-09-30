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
        - counts (list): Number of pixels inside a bin. Returns probs or counts.
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
        counts = [c / (n * width) for c in counts]
    return counts, edges

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
    counts, edges = histogram(data, n_bins=n_bins, data_range=data_range, probs=probs)

    # Draw bars
    widths = [edges[i+1] - edges[i] for i in range(len(edges)-1)]
    lefts  = edges[:-1]

    fig, ax = plt.subplots()
    ax.bar(lefts, counts, width=widths, align="edge")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density" if probs else "Count")
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=200)
        
    plt.show()
    return outfile