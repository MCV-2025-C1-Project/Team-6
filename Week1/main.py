import argparse
import numpy as np

from metrics import mean_average_precision
from pathlib import Path
from io_utils import read_images, read_pickle, plot_query_results
from descriptors import compute_descriptors
from similarity_measures import compute_similarities


# TODO: Save results in a pickle file called result (then rename it for each method used)
# NOTE: WE CAN ADD MORE ARGUMENTS IN THE DATA PARSER TO ACCOUNT FOR THE 2 STRATEGIES TO USE, OR WE CAN MAKE 
# COMPUTE DESCRIPTORS TO DO WHATEVER, THIS IS A FIRST SKELETON


def main(data_dir: Path, k_results: int = 5) -> None:
    # Read query images
    images = read_images(data_dir) 

    # Obtain database descriptors.
    bbdd_descriptors = read_pickle("BBDD/BBDD_descriptors_rgb.pkl")

    # Compute query images descriptors
    img_descriptors = compute_descriptors(images)

    # Compute similarities
    similarities = compute_similarities(img_descriptors, bbdd_descriptors['descriptors'])

    # Sort the similarities
    sorted_similarities = np.sort(similarities, axis=1)

    # Extract indices and similarity values from sorted tuples
    all_predictions = []
    results_indices = []
    results_similarities = []
    
    for i in range(sorted_similarities.shape[0]):
        # Extract both indices and similarity values
        indices = [t[1] for t in sorted_similarities[i]]
        sim_values = [t[0] for t in sorted_similarities[i]]
        
        all_predictions.append(indices)
        results_indices.append(indices[:k_results])
        results_similarities.append(sim_values[:k_results])

    print("Most similar images for each query:")
    for i, (indices, sim_values) in enumerate(zip(results_indices, results_similarities)):
        print(f"Query {i}:")
        for j, (idx, sim_val) in enumerate(zip(indices, sim_values)):
            print(f"  Result {j+1}: idx={idx}, sim={sim_val:.4f}")
        print()

    # Plot the results with similarity values
    plot_query_results(images, results_indices, results_similarities, k=k_results, 
                      save_path=data_dir / "query_results_plot.png")

    # Compute MAP score
    gt = read_pickle(data_dir / "gt_corresps.pkl")
    map_score = mean_average_precision(all_predictions, gt)
    print(f"MAP@K score: {map_score:.4f}")


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

    # Check directory
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory.")

    # Process dataset
    main(data_dir)