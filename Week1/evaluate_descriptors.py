"""Evaluates retrieval on query images by loading precomputed BBDD descriptors, computing query descriptors
and ranking with multiple metrics."""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from params import experiments
from metrics import mean_average_precision
from pathlib import Path
from io_utils import read_images, read_pickle
from descriptors import compute_descriptors
from similarity_measures import compute_similarities


# NOTE: WE CAN ADD MORE ARGUMENTS IN THE DATA PARSER TO ACCOUNT FOR THE 2 STRATEGIES TO USE, OR WE CAN MAKE 
# COMPUTE DESCRIPTORS TO DO WHATEVER, THIS IS A FIRST SKELETON


def main(data_dir: Path, k_results: int = 5) -> None:
    # Read query images
    images = read_images(data_dir) 

    # Obtain database descriptors.
    
    methods =  experiments["methods"]
    metrics = experiments["metrics"]
    bins = experiments["n_bins"]

    mapk_scores = {}
    k= 5
    for method in methods:
        for n_bins in bins:
            bbdd_descriptors = read_pickle(Path(__file__).resolve().parent / "descriptors" / f"{method}_{n_bins}bins_descriptors.pkl")

            img_descriptors = compute_descriptors(images, method=method, n_bins=n_bins, save_pkl=False)
            for metric in metrics:

                print(f"Computing similarities using {metric} metric.")
                similarities = compute_similarities(img_descriptors, bbdd_descriptors, metric=metric)
                indices = np.argsort(similarities, axis=1)

                gt = read_pickle(data_dir / "gt_corresps.pkl")

                # Compute MAP score
                map_score = mean_average_precision(indices, gt, k)

                print(f"MAP@K score: {map_score:.4f}, using {metric} metric, {method} descriptors with {n_bins} bins.")
                mapk_scores[f"{method}_{n_bins}bins_{metric}"] = map_score
            
            print()
        print()

    for bin in bins:
    # Build matrix of scores
        score_matrix = []
        for method in methods:
            row = []
            for metric in metrics:
                row.append(mapk_scores[f"{method}_{bin}bins_{metric}"])
            score_matrix.append(row)

        score_matrix = np.array(score_matrix)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(score_matrix, cmap="viridis")

        # Show all ticks and label them
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.set_yticklabels(methods)

        # Add values on top of cells
        for i in range(len(methods)):
            for j in range(len(metrics)):
                ax.text(j, i, f"{score_matrix[i, j]:.3f}", ha="center", va="center", color="w")

        ax.set_title(f"MAP@5 scores by descriptor and metric for {bin} bins")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"mapk_matrix_{bin}.png")
        plt.close()


    print("Top 5 Configurations:")
    sorted_mapk = dict(sorted(mapk_scores.items(), key=lambda item: item[1], reverse=True))
    for i, (key, value) in enumerate(sorted_mapk.items()):
        if i < 5:
            print(f"  {key}: {value:.4f}")
        else:
            break

    print("Writing results to txt file.")
    with open("results.txt", "w") as f:
        for key, value in mapk_scores.items():
            f.write(f"{key}: {value:.4f}\n")

if __name__ == "__main__":

    # Parse data directory argument (by default the dev set is used)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir',
        type=Path,
        default=Path(__file__).resolve().parent / "qsd1_w1",
        help='Path to the dataset directory.'
    )
    data_dir = parser.parse_args().data_dir
    print(f"Using data from {data_dir}.")
    # Check directory
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory.")

    # Process dataset
    main(data_dir)