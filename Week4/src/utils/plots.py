from pathlib import Path
from typing import Optional, List
import numpy as np
import cv2

import matplotlib
# matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
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



def _vis_compat(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make two images stackable: same size, 3 channels, same dtype."""
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)

    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if b.ndim == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)

    if a.dtype != b.dtype:
        b = b.astype(a.dtype)
    return a, b


def plot_image_comparison(den_images: List[np.ndarray], og_images: List[np.ndarray], max_imgs: int=None) -> None:
    for i, (denoised, original) in enumerate(zip(den_images, og_images)):
        if max_imgs is not None and i >= max_imgs:
            break
        img_vis, den_vis = _vis_compat(original, denoised)
        final = np.hstack([img_vis, den_vis])
    
        cv2.imshow('Original vs denoised image', final)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def plot_all_comparisons(
    og_images: List[np.ndarray], 
    den_images: List[np.ndarray], 
    gt_images: List[np.ndarray], 
    scores: List[float], 
    save_path: Optional[Path] = None
) -> None:
    """
    Plots all original, denoised, and ground truth images in a single grid
    and saves it to a file.
    
    Args:
        og_images: List of original noisy images.
        den_images: List of denoised images.
        gt_images: List of ground truth images.
        scores: List of scores corresponding to each image.
        save_path: Path to save the combined plot.
    """
    n_images = len(og_images)
    if n_images == 0:
        print("No images to plot.")
        return

    # Create a grid of N rows and 3 columns
    fig, axes = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))
    
    # Handle case of n_images = 1, where axes is 1D
    if n_images == 1:
        axes = np.array([axes]) # Reshape to (1, 3)
        
    for i in range(n_images):
        # Get images and score
        og, den, gt = og_images[i], den_images[i], gt_images[i]
        score = scores[i]
        
        # Make images compatible for visualization
        # og_viz will be the reference
        og_viz, den_viz = _vis_compat(og, den)
        og_viz, gt_viz = _vis_compat(og_viz, gt) # gt_viz resized to og_viz

        # Plot Original
        ax_og = axes[i, 0]
        ax_og.imshow(og_viz)
        ax_og.set_title(f"Image {i}: Original (Noisy)")
        ax_og.axis('off')
        
        # Plot Denoised
        ax_den = axes[i, 1]
        ax_den.imshow(den_viz)
        ax_den.set_title(f"Denoised (Score: {score:.4f})")
        ax_den.axis('off')

        # Plot Ground Truth
        ax_gt = axes[i, 2]
        ax_gt.imshow(gt_viz)
        ax_gt.set_title(f"Image {i}: Ground Truth")
        ax_gt.axis('off')
    
    plt.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show() # Fallback to showing the plot if no path is given
    
    plt.close(fig) # Free up memory