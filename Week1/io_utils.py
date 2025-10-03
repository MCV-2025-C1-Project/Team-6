import pickle
from typing import Any, List
import cv2
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# Read data from pickle file
def read_pickle(file_path: Path) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


# Write data to a pickle file
def write_pickle(data: Any, file_path: Path) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def read_images(dir_path: Path) -> List[np.ndarray]:
    """
    Reads all JPG images from the given directory using OpenCV (sorted by their filenames).

    Args:
        dir_path (Path):    Path to the image directory.

    Returns:
        List[np.ndarray]:   A list containing the images.
    """
    return [
        cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        for img_path in sorted(dir_path.glob("*.jpg"))
    ]



def plot_query_results(queries, results, similarity_values, k=5, save_path=None):
    """
    Plot query images and their k most similar results
    
    Args:
        queries: list of query images
        results: list of lists where each sublist contains k most similar image indices
        similarity_values: list of lists with corresponding similarity values
        k: number of results to show per query
        save_path: path to save the plot (if None, shows the plot)
    """
    n_queries = len(queries)
    
    # Create figure
    fig, axes = plt.subplots(n_queries, k + 1, figsize=(15, 3 * n_queries))
    
    # If only one query, make axes 2D
    if n_queries == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_queries):
        # Plot query image
        axes[i, 0].imshow(queries[i])
        axes[i, 0].set_title(f'Query {i}')
        axes[i, 0].axis('off')
        
        # Plot k most similar images
        for j, (img_idx, sim_value) in enumerate(zip(results[i][:k], similarity_values[i][:k])):
            img_path = f"BBDD/bbdd_{img_idx:05d}.jpg"
            try:
                bbdd_img = plt.imread(img_path)
                axes[i, j + 1].imshow(bbdd_img)
                axes[i, j + 1].set_title(f'Result {j+1}\nidx: {img_idx}\nsim: {sim_value:.4f}', 
                                        fontsize=10)
            except FileNotFoundError:
                axes[i, j + 1].text(0.5, 0.5, f'Image {img_idx}\nnot found\nsim: {sim_value:.4f}', 
                                   ha='center', va='center', transform=axes[i, j + 1].transAxes,
                                   fontsize=10)
            axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()