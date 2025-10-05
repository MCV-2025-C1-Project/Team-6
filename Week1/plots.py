from pathlib import Path
from typing import Optional, List
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt

def plot_query_results(queries: List[np.ndarray], 
                       results: List[np.ndarray], 
                       similarity_values: np.ndarray, 
                       k: int = 5, 
                       save_path: Optional[Path] = None) -> None:
    """
    Plot query images and their k most similar results.
    
    Args:
        queries: List of query images.
        results: List of lists where each sublist contains k most similar image indices.
        similarity_values: Array with corresponding similarity values of shape (len(queries), len(bbdd)).
        k: Number of results to show per query.
        save_path: Path to save the plot (if None, shows the plot).
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
        for j, (img_idx, sim_value) in enumerate(zip(results[i], similarity_values[i])):
            img_path = Path(__file__).resolve().parent.parent / "BBDD" / f"bbdd_{img_idx:05d}.jpg" # TODO: is this a correct relative path? DIEGO
            try:
                bbdd_img = plt.imread(str(img_path))
                axes[i, j + 1].imshow(bbdd_img)
                axes[i, j + 1].axis('off')
                axes[i, j + 1].set_title(f'Rank {j+1}\nidx: {img_idx}\nsim: {sim_value:.4f}', 
                                        fontsize=10)
            except FileNotFoundError:
                axes[i, j + 1].text(0.5, 0.5, f'Image {img_idx}\nnot found\nsim: {sim_value:.4f}', 
                                   ha='center', va='center', transform=axes[i, j + 1].transAxes,
                                   fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_descriptors_difference(query_descriptors: List[np.ndarray], 
                                most_similar_descriptors: List[np.ndarray], 
                                save_path: Optional[Path] = None) -> None:
    """
    Plot the difference between two image descriptors for the whole query set.
    
    Args:
        query_descriptors: List of np.ndarray, each of shape (D,) representing query image descriptors.
        most_similar_descriptors: List of np.ndarray, each of shape (D,) representing the most similar image descriptors.
        save_path: Optional path to save the plot. If None, the plot is shown.
    """

    n_queries = len(query_descriptors)
    
    fig, axes = plt.subplots(n_queries, 1, figsize=(14, 4 * n_queries))
    
    # To list if only one query
    if n_queries == 1:
        axes = [axes]
    
    # Plotting comparisons
    for i in range(n_queries):
        ax = axes[i]
        
        # Convert to numpy arrays if they aren't already
        query_desc = np.asarray(query_descriptors[i])
        similar_desc = np.asarray(most_similar_descriptors[i])
        
        ax.plot(query_desc, label='Query Descriptor', color='blue', alpha=0.8, linewidth=1.5)
        ax.plot(similar_desc, label='BBDD Descriptor', color='orange', alpha=0.8, linewidth=1.5)
        
        ax.set_title(f'Descriptor Comparison - Query {i}', fontsize=14, pad=20)
        ax.set_xlabel('Descriptor Dimension', fontsize=12)
        ax.set_ylabel('Descriptor Value', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
