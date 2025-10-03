import argparse
import numpy as np

from metrics import mean_average_precision
from pathlib import Path
from io_utils import read_images, read_pickle
from descriptors import compute_descriptors
from similarity_measures import compute_similarities
from typing import Optional, Dict, Any


# TODO: Save results in a pickle file called result (then rename it for each method used)
# NOTE: WE CAN ADD MORE ARGUMENTS IN THE DATA PARSER TO ACCOUNT FOR THE 2 STRATEGIES TO USE, OR WE CAN MAKE 
# COMPUTE DESCRIPTORS TO DO WHATEVER, THIS IS A FIRST SKELETON

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

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
        for j, (img_idx, sim_value) in enumerate(zip(results[i], similarity_values[i])):
            img_path = Path(__file__).resolve().parent.parent / "BBDD" / f"bbdd_{img_idx:05d}.jpg" # TODO: is this a correct relative path? DIEGO
            try:
                bbdd_img = plt.imread(img_path)
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


def main(
    data_dir: Path, 
    pickle_path: Path,
    k: int = 5,
    distance_metric: str = 'euclidean',
    plot_file: str = 'query_rgb_plot.png',
    descriptor: str = 'rgb', 
    descriptor_params: Optional[Dict[str, Any]] = None) -> None:

    # Read query images
    images = read_images(data_dir) 

    # Obtain database descriptors.
    bbdd_descriptors = read_pickle(pickle_path)

    # Compute query images descriptors
    # This compute does NOT save any .pkl file
    img_descriptors = compute_descriptors(imgs=images, method=descriptor, params=descriptor_params) 

    # Compute similarities
    similarities = compute_similarities(img_descriptors, bbdd_descriptors['descriptors'], metric=distance_metric)

    # Sort the similarities and obtain their indices
    indices = np.argsort(similarities, axis=1)
    sorted_sims = np.take_along_axis(similarities, indices, axis=1)

    # Extract the best k results
    results_indices = indices[:, :k]
    results_similarities = sorted_sims[:, :k]

    print(results_indices)

    # Load ground truth
    gt = read_pickle(data_dir / "gt_corresps.pkl")
    
    # Print some results
    print("Most similar images for each query:")
    for i, (res_idx, sim_values) in enumerate(zip(results_indices, results_similarities)):
        print(f"Query {i} - GT: {gt[i]}:")
        for j, (idx, sim_val) in enumerate(zip(res_idx, sim_values)):
            print(f"  Result {j+1}: idx={idx}, sim={sim_val:.4f}")
        print()
 
    # Plot the results with similarity values
    plot_query_results(images, results_indices, results_similarities, k=k, 
                      save_path=data_dir / plot_file)

    # Compute MAP score
    map_score = mean_average_precision(indices, gt, k)
    print(f"MAP@K score: {map_score:.4f}")


if __name__ == "__main__":

    # TODO: Change relatives paths if possible
    # TODO: Check descriptor params

    # Parse data directory argument (by default the dev set is used)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir',
        type=Path,
        default=Path(__file__).resolve().parent / "qsd1_w1",
        help='Path to the dataset directory.'
    )
    parser.add_argument(
        '-k', '--k-results',
        type=int,
        default=5,
        help='K-results values for mAP@k calculation.'
    )
    parser.add_argument(
        '-pkl', '--pickle-path',
        type=Path,
        default=Path(__file__).resolve().parent.parent / "BBDD" / "BBDD_descriptors_rgb.pkl",
        help='Path to the pickle file containing the database descriptors.'
    )
    parser.add_argument(
        '-dist', '--distance-metric',
        type=str,
        default='euclidean',
        help='Distance metric to use: euclidean, l1, chi2, histogram_intersection, hellinger, cosine, bhattacharyya'
    )
    parser.add_argument(
        '-plt', '--plot-file',
        type=str,
        default='query_rgb_plot.png',
        help='Path to the plot file for saving the query results.'
    )

    # TODO: Pending to test
    parser.add_argument(
        '-desc', '--descriptor',
        type=str,
        default='rgb',
        help='Descriptor to use: rgb, hsv, hs_rgb, hs'
    )
    parser.add_argument(
        '-params', '--descriptor-params',
        type=str,
        default='{}',
        help='Descriptor parameters as a dictionary string.'
    )
    
    data_dir = parser.parse_args().data_dir
    k_results = parser.parse_args().k_results
    pickle_path = parser.parse_args().pickle_path
    distance_metric = parser.parse_args().distance_metric
    plot_file = parser.parse_args().plot_file
    descriptor = parser.parse_args().descriptor
    descriptor_params = parser.parse_args().descriptor_params
    
    # Check directory
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory.")
    # Check k-results
    if k_results <= 0 or k_results > 20:
        raise ValueError(f"{k_results} is not a valid k-results value.")
    # Check pickle path
    if not pickle_path.is_file():
        raise ValueError(f"{pickle_path} is not a valid file.")
    # Check plot file path
    if not plot_file.endswith('.png'):
        raise ValueError(f"{plot_file} is not a valid PNG file.")
    # Check distance metric
    valid_metrics = ['euclidean', 'l1', 'chi2', 'histogram_intersection', 'hellinger', 'cosine', 'bhattacharyya']
    if distance_metric not in valid_metrics:
        raise ValueError(f"{distance_metric} is not a valid metric. Choose from {valid_metrics}.")
    # Check descriptor
    valid_descriptors = ['rgb', 'hsv', 'hs_rgb', 'hs']
    if descriptor not in valid_descriptors:
        raise ValueError(f"{descriptor} is not a valid descriptor. Choose from {valid_descriptors}.")

    # Process dataset
    main(data_dir=data_dir, pickle_path=pickle_path, k=k_results, distance_metric=distance_metric)