"""Evaluates retrieval on query images by loading precomputed BBDD descriptors, computing query descriptors
and ranking with multiple metrics."""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from params import experiments
from metrics import mean_average_precision
from pathlib import Path
from io_utils import read_images, read_pickle
from piramid_descriptors import compute_spatial_descriptors
from similarity_measures import compute_similarities
from plots import plot_descriptors_difference, plot_query_results

def main(data_dir: Path, generate_plots=False) -> None:

    # Create dir for outputs
    os.makedirs(Path(__file__).resolve().parent / 'outputs', exist_ok=True)

    # Read query images
    images = read_images(data_dir) 

    # Read experiment parameters
    methods = experiments["methods"]
    metrics = experiments["metrics"]
    bins = experiments["n_bins"]
    k_values = experiments["k_values"]
    n_crops = experiments["n_crops"]
    # Evaluation
    
    for k in k_values:
        mapk_scores = {}
        for n_crop in n_crops:
            for method in methods:
                for n_bins in bins:
                    bbdd_descriptors = read_pickle(Path(__file__).resolve().parent / "descriptors" / f"{method}_{n_bins}bins_{n_crop}crops_noWeights_descriptors.pkl")
                    img_descriptors = compute_spatial_descriptors(images, method=method, n_bins=n_bins, save_pkl=False, n_crops=n_crop)

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

                        if generate_plots:
                            # Plot the descriptors difference with the most similar
                            best_descriptors = [bbdd_descriptors[idx] for idx in results_indices[:, 0]]
                            plot_descriptors_difference(img_descriptors, best_descriptors,
                                        save_path=Path(__file__).resolve().parent / 'outputs' / f'descriptor_difference_at{k}_{method}_{n_bins}_{metric}_{n_crop}.png')
                            
                            # Plot the results with similarity values
                            plot_query_results(images, results_indices, results_similarities, k=k, 
                                        save_path=Path(__file__).resolve().parent / 'outputs' / f'query_at{k}_{method}_{n_bins}_{metric}_{n_crop}.png')

                        # Compute MAP score
                        map_score = mean_average_precision(indices, gt, k)

                        print(f"MAP@{k} score: {map_score:.4f}, using {metric} metric, {method} descriptors with {n_bins} bins.")
                        mapk_scores[f"{method}_{n_bins}bins_{metric}_{n_crop}crops"] = map_score
                    
                    print()
                print()

            # Build matrices for each method-metric pair across bins and crops
        matrix = False
        if matrix:
            for method in methods:
                for metric in metrics:
                    # Create matrix rows = n_crops, cols = bins
                    score_matrix = np.zeros((len(n_crops), len(bins)))
                    
                    for i, n_crop in enumerate(n_crops):
                        for j, n_bin in enumerate(bins):
                            key = f"{method}_{n_bin}bins_{metric}_{n_crop}crops"
                            if key in mapk_scores:
                                score_matrix[i, j] = mapk_scores[key]
                            else:
                                score_matrix[i, j] = np.nan  # in case something missing

                    # Plot heatmap
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(score_matrix, cmap="viridis", aspect="auto")

                    # Set labels
                    ax.set_xticks(np.arange(len(bins)))
                    ax.set_yticks(np.arange(len(n_crops)))
                    ax.set_xticklabels(bins)
                    ax.set_yticklabels(n_crops)
                    ax.set_xlabel("Number of bins")
                    ax.set_ylabel("Number of crops")
                    ax.set_title(f"MAP@{k} for {method} - {metric}")

                    # Annotate cells with values
                    for i in range(len(n_crops)):
                        for j in range(len(bins)):
                            val = score_matrix[i, j]
                            if not np.isnan(val):
                                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="w", fontsize=8)

                    # Add colorbar
                    fig.colorbar(im, ax=ax)

                    plt.tight_layout()
                    plt.savefig(
                        Path(__file__).resolve().parent / "outputs" / f"map{k}_matrix_{method}_{metric}_noWeights.png"
                    )
                    plt.close()
        else:
            for method in methods:
                for metric in metrics:
                    
                    # Plot function (line plot)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Iterate over the values of bins to create separate lines
                    for j, n_bin in enumerate(bins):
                        # Create a list to hold the scores for the current n_bin across all n_crops
                        scores_for_n_bin = []
                        
                        # Iterate over n_crops (the x-axis)
                        for i, n_crop in enumerate(n_crops):
                            key = f"{method}_{n_bin}bins_{metric}_{n_crop}crops"
                            if key in mapk_scores:
                                score = mapk_scores[key]
                            else:
                                score = np.nan # Use NaN if data is missing
                            
                            scores_for_n_bin.append(score)

                        # Prepare data for plotting
                        x_data = np.array(n_crops) # The x-axis data
                        y_data = np.array(scores_for_n_bin) # The score data
                        
                        # Find valid data points (not NaN)
                        valid_indices = ~np.isnan(y_data)
                        
                        # Plot the line for the current n_bin
                        ax.plot(
                            x_data[valid_indices], 
                            y_data[valid_indices], 
                            marker='o', 
                            linestyle='-', 
                            label=f"{n_bin} bins"
                        )

                    # Set labels and title
                    ax.set_xlabel("Number of crops") # Changed X-axis label
                    ax.set_ylabel(f"MAP@{k} Score")
                    ax.set_title(f"MAP@{k} Score vs. Crops for {method} - {metric}")
                    
                    # Set x-ticks to correspond to the actual n_crops numbers
                    ax.set_xticks(n_crops)
                    
                    # Add a legend to distinguish the lines (different number of bins)
                    ax.legend(title="Number of Bins")

                    # Add grid for better readability
                    ax.grid(True, linestyle='--', alpha=0.7)

                    plt.tight_layout()
                    plt.savefig(
                        Path(__file__).resolve().parent / "outputs" / f"map{k}_function_crops_xaxis_{method}_{metric}_noWeights.png"
                    )
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