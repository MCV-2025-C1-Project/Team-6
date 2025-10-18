from pathlib import Path
from typing import Optional, List
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

def plot_query_results(queries: List[np.ndarray], 
    results: List[np.ndarray], 
    similarity_values: np.ndarray, 
    k: int = 5, 
    save_path: Optional[Path] = None
) -> None:
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
            img_path = SCRIPT_DIR.parent.parent.parent / "BBDD" / f"bbdd_{img_idx:05d}.jpg"  
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

    plt.close()


def plot_segmentation_results(queries: List[np.ndarray], 
    cropped_images: List[np.ndarray], 
    results: List[np.ndarray], 
    similarity_values: np.ndarray, 
    k: int = 5, 
    save_path: Optional[Path] = None
) -> None:
    
    """
    Plot query images with background, their cropped versions, and their k most similar results.
    
    Args:
        queries: List of query images with background.
        cropped_images: List of cropped query images (paintings).
        results: List of lists where each sublist contains k most similar image indices.
        similarity_values: Array with corresponding similarity values of shape (len(queries), len(bbdd)).
        k: Number of results to show per query.
        save_path: Path to save the plot (if None, shows the plot).
    """

    n_queries = len(queries)
    
    # Create figure
    fig, axes = plt.subplots(n_queries, k + 2, figsize=(18, 3 * n_queries))
    
    # If only one query, make axes 2D
    if n_queries == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_queries):
        # Plot query image with background
        axes[i, 0].imshow(queries[i])
        axes[i, 0].set_title(f'Query {i} (with BG)')
        axes[i, 0].axis('off')
        
        # Plot cropped painting
        axes[i, 1].imshow(cropped_images[i])
        axes[i, 1].set_title(f'Query {i} (cropped)')
        axes[i, 1].axis('off')
        
        # Plot k most similar images
        for j, (img_idx, sim_value) in enumerate(zip(results[i], similarity_values[i])):
            img_path = SCRIPT_DIR.parent.parent.parent / "BBDD" / f"bbdd_{img_idx:05d}.jpg"  
            try:
                bbdd_img = plt.imread(str(img_path))
                axes[i, j + 2].imshow(bbdd_img)
                axes[i, j + 2].axis('off')
                axes[i, j + 2].set_title(f'Rank {j+1}\nidx: {img_idx}\nsim: {sim_value:.4f}', 
                                        fontsize=10)
            except FileNotFoundError:
                axes[i, j + 2].text(0.5, 0.5, f'Image {img_idx}\nnot found\nsim: {sim_value:.4f}', 
                                   ha='center', va='center', transform=axes[i, j + 2].transAxes,
                                   fontsize=10)
    plt.tight_layout()  

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


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


def show_split_debug(
    image: np.ndarray,
    parts: List[np.ndarray],
    masks: List[np.ndarray],
    mask_final: np.ndarray
) -> None:
    """
    Displays a debug visualization of the image split results.

    This function creates a matplotlib plot showing:
      - The original image (with a vertical red cut line if split).
      - Part 1 overlaid with its mask.
      - Part 2 overlaid with its mask (if a 2-part split occurred).
      - The final reconstructed mask.

    The plot is displayed interactively using plt.show().

    Args:
        image: The original, full input image.
        parts: A list of image parts (e.g., [left_img, right_img] if
            split, or [original_img] if not).
        masks: A list of masks, one corresponding to each part in `parts`.
        mask_final: The final, full-size mask reconstructed from the
            individual part masks.
    """
    # Find the cut location (it's the width of the left part)
    cut_x = parts[0].shape[1] if len(parts) == 2 else None

    ncols = 4 if len(parts) == 2 else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    
    # Ensure axes is always an array, even if ncols=1 (though it's 2 or 4 here)
    if not isinstance(axes, (np.ndarray, list)):
        axes = [axes]

    # Original (with cut line if applicable) ---
    axes[0].imshow(image)
    axes[0].set_title("Original")
    if cut_x is not None:
        axes[0].axvline(cut_x, linewidth=2, color='r')
    axes[0].axis("off")

    # Part 1 + mask ---
    axes[1].imshow(parts[0])
    axes[1].imshow(masks[0], alpha=0.4) # Overlay mask with transparency
    axes[1].set_title("Part 1 + mask")
    axes[1].axis("off")

    if len(parts) == 2:
        # Part 2 + mask ---
        axes[2].imshow(parts[1])
        axes[2].imshow(masks[1], alpha=0.4)
        axes[2].set_title("Part 2 + mask")
        axes[2].axis("off")

        # Reconstructed mask ---
        axes[3].imshow(mask_final, cmap="gray")
        axes[3].set_title("Reconstructed mask")
        axes[3].axis("off")
    else:
        # Final mask (when no split occurred) ---
        axes[1].imshow(mask_final, cmap="gray")
        axes[1].set_title("Final mask")
        axes[1].axis("off")

    plt.tight_layout()
    plt.show() # Display the plot