import argparse

import numpy as np

from pathlib import Path
from io_utils import read_images, read_pickle, write_pickle
from descriptors import compute_descriptors
from similarity_measures import compute_similarities
from params import best_config1, best_config2

SCRIPT_DIR = Path(__file__).resolve().parent

def main(
    data_dir: Path,
    method: str = 'rgb',
    distance_metric: str = 'euclidean',
    n_bins: int = 32) -> None:

    # Read query images
    images = read_images(data_dir)

    # Obtain/compute database images descriptors
    try:
        bbdd_descriptors = read_pickle(SCRIPT_DIR / "descriptors" / f"{method}_{n_bins}bins_descriptors.pkl")
    except FileNotFoundError:
        bbdd_images = read_images(SCRIPT_DIR.parent.parent / "BBDD")
        bbdd_descriptors = compute_descriptors(bbdd_images, method, n_bins)

    # Compute query images descriptors
    descriptors = compute_descriptors(images, method, n_bins) 

    # Compute similarities
    similarities = compute_similarities(descriptors, bbdd_descriptors, distance_metric)

    # Sort the indices resulting from the similarities sorting
    indices = np.argsort(similarities, axis=1)

    # Save results for k=10
    k = 10
    results = indices[:, :k].astype(int).tolist()
    write_pickle(results, SCRIPT_DIR / f"{method}_{n_bins}bins_{distance_metric}_MAP@{k}.pkl")


if __name__ == "__main__":

    # Parse data directory argument (by default the test set is used)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir',
        type=Path,
        default=SCRIPT_DIR.parent / "qst1_w1",
        help='Path to the dataset directory.'
    )
    data_dir = parser.parse_args().data_dir

    # Check directory
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory.")
    
    # Parameters for the chosen methods
    methods = [best_config1["method"], best_config2["method"]]
    metrics = [best_config1["metric"], best_config2["metric"]]
    n_bins = [best_config1["n_bins"], best_config2["n_bins"]]

    # Run results for both methods
    for method, metric, n_bins in zip(methods, metrics, n_bins):
        main(data_dir, method=method, distance_metric=metric, n_bins=n_bins)

    print("Results generated successfully.")
