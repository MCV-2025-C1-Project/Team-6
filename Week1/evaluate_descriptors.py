"""Evaluates retrieval on query images by loading precomputed BBDD descriptors, computing query descriptors
and ranking with multiple metrics."""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from params import experiments, best_config_1, best_config_5
from metrics import mean_average_precision
from pathlib import Path
from io_utils import read_images, read_pickle
from descriptors import compute_descriptors
from similarity_measures import compute_similarities
from plots import plot_query_results, plot_descriptors_difference


# NOTE: WE CAN ADD MORE ARGUMENTS IN THE DATA PARSER TO ACCOUNT FOR THE 2 STRATEGIES TO USE, OR WE CAN MAKE 
# COMPUTE DESCRIPTORS TO DO WHATEVER, THIS IS A FIRST SKELETON


def main(data_dir: Path) -> None:

    # Create dir for outputs
    os.makedirs(Path(__file__).resolve().parent / 'outputs', exist_ok=True)

    # Read query images
    images = read_images(data_dir) 

    methods = experiments["methods"]
    metrics = experiments["metrics"]
    bins = experiments["n_bins"]
    k = experiments["k_value"]

    # methods = best_config_5["methods"]
    # metrics = best_config_5["metrics"]
    # bins = best_config_5["n_bins"]
    # k = best_config_5["k_value"]

    mapk_scores = {}
    for method in methods:
        for n_bins in bins:
            bbdd_descriptors = read_pickle(Path(__file__).resolve().parent / "descriptors" / f"{method}_{n_bins}bins_descriptors.pkl")

            img_descriptors = compute_descriptors(images, method=method, n_bins=n_bins, save_pkl=False)
            for metric in metrics:

                print(f"Computing similarities using {metric} metric.")
                similarities = compute_similarities(img_descriptors, bbdd_descriptors, metric=metric)
                
                # Sort the similarities and obtain their indices
                indices = np.argsort(similarities, axis=1)
                sorted_sims = np.take_along_axis(similarities, indices, axis=1)

                # Extract the best k results
                results_indices = indices[:, :k]

                results_similarities = sorted_sims[:, :k]

                # Load ground truth
                gt = read_pickle(data_dir / "gt_corresps.pkl")

                # Print some results
                print(f"Most similar images for each query at K={k}:")
                for i, (res_idx, sim_values) in enumerate(zip(results_indices, results_similarities)):
                    print(f"Query {i} - GT: {gt[i]}:")
                    for j, (idx, sim_val) in enumerate(zip(res_idx, sim_values)):
                        print(f"  Result {j+1}: idx={idx}, sim={sim_val:.4f}")
                    print()
                
                gt = read_pickle(data_dir / "gt_corresps.pkl")

                
                # Plot the descriptors difference with the most similar
                # best_descriptors = [bbdd_descriptors[idx] for idx in results_indices[:, 0]]
                # plot_descriptors_difference(img_descriptors, best_descriptors,
                #             save_path=Path(__file__).resolve().parent / 'outputs' / f'descriptor_difference_at{k}_{method}_{n_bins}_{metric}.png')
                
                # # Plot the results with similarity values
                # plot_query_results(images, results_indices, results_similarities, k=k, 
                #                 save_path=Path(__file__).resolve().parent / 'outputs' / f'query_at{k}_{method}_{n_bins}_{metric}.png')

                # Compute MAP score
                map_score = mean_average_precision(indices, gt, k)

                print(f"MAP@{k} score: {map_score:.4f}, using {metric} metric, {method} descriptors with {n_bins} bins.")
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

        ax.set_title(f"MAP@{k} scores by descriptor and metric for {bin} bins")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(Path(__file__).resolve().parent / 'outputs' / f"map{k}_matrix_{bin}.png")
        plt.close()


    print(f"Top 5 Configurations at K={k}:")
    sorted_mapk = dict(sorted(mapk_scores.items(), key=lambda item: item[1], reverse=True))
    for i, (key, value) in enumerate(sorted_mapk.items()):
        if i < 5:
            print(f"  {key}: {value:.4f}")
        else:
            break

    print("Writing results to txt file.")
    with open(Path(__file__).resolve().parent / 'outputs' / f"results_at{k}.txt", "w") as f:
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