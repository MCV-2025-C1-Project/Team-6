import argparse
from pathlib import Path
import os

import numpy as np

from evaluations.similarity_measures import compute_similarities
from evaluations.metrics import mean_average_precision
from descriptors import compute_spatial_descriptors
from utils.io_utils import read_images, read_pickle
from utils.plots import plot_segmentation_results
from background import apply_segmentation, crop_images
from params import best_config_segmentation, best_config_descriptors

SCRIPT_DIR = Path(__file__).resolve().parent

def main(dir1: Path, dir2: Path, k: int = 10) -> None:

    # Descriptor parameters
    desc_params = best_config_descriptors
    print("Descriptor parameters:", desc_params)

    # Segmentation parameters
    segm_params = best_config_segmentation
    print("Segmentation parameters:", segm_params)

    # Obtain/compute database images descriptors
    try:
        print("Loading database descriptors...")
        bbdd_descriptors = read_pickle(SCRIPT_DIR / "descriptors" / f"{desc_params['color_space']}_{desc_params['n_bins']}bins_{desc_params['n_crops']}crops_noPyramid_descriptors.pkl")
    except FileNotFoundError:
        print("Unable to load database descriptors. Computing them...")
        bbdd_images = read_images(SCRIPT_DIR.parent.parent / "BBDD")
        bbdd_descriptors = compute_spatial_descriptors(bbdd_images, method=desc_params['color_space'], n_bins=desc_params['n_bins'], pyramid=False, n_crops=desc_params['n_crops'], save_pkl=True)


    """Process dataset of images without background. Task1 and Task2."""
    print("Processing development set without background...")

    # Read query images
    images1 = read_images(dir1)

    # Compute query images descriptors
    descriptors = compute_spatial_descriptors(images1, method=desc_params["color_space"], n_bins=desc_params["n_bins"], pyramid=False, n_crops=desc_params["n_crops"])

    # Compute similarities
    similarities = compute_similarities(descriptors, bbdd_descriptors, desc_params["metric"])

    # Sort the indices resulting from the similarities sorting
    indices = np.argsort(similarities, axis=1)

    # Results for k best matches
    results = indices[:, :k].astype(int).tolist()
    print("Results development set (noBG):", results)

    # Compute MAP score
    gt = read_pickle(dir1 / "gt_corresps.pkl")
    print("Ground truths:", gt)
    map_score = mean_average_precision(indices, gt, k)
    print(f"MAP@{k} score: {map_score:.4f}")

    """Process dataset of images with background."""
    print("Processing development set with background...")

    # Read query images
    images2 = read_images(dir2)

    # Detect BG from images
    masks = apply_segmentation(images2, segm_params)

    # Crop paintings
    paintings = crop_images(images2, masks)

    # Compute query images descriptors
    descriptors = compute_spatial_descriptors(paintings, method=desc_params["color_space"], n_bins=desc_params["n_bins"], pyramid=False, n_crops=desc_params["n_crops"])

    # Compute similarities
    similarities = compute_similarities(descriptors, bbdd_descriptors, desc_params["metric"])

    # Sort the indices resulting from the similarities sorting
    indices = np.argsort(similarities, axis=1)

    # Results for k best matches
    results = indices[:, :k].astype(int).tolist()
    print("Results development set (BG):", results)

    # Compute MAP score
    gt = read_pickle(dir2 / "gt_corresps.pkl")
    print("Ground truths:", gt)
    map_score = mean_average_precision(indices, gt, k)
    print(f"MAP@{k} score: {map_score:.4f}")

    # Make directory if not setted up
    os.makedirs(SCRIPT_DIR / "results", exist_ok=True)

    # Plot results
    plot_segmentation_results(queries=images2,
                            cropped_images=paintings,
                            results=indices[:, :k],
                            similarity_values=np.take_along_axis(similarities, indices[:, :k], axis=1),
                            k=k,
                            save_path=SCRIPT_DIR / "results" / "segmentation_results.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dir1', '--data-dir1',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd1_w2",
        help='Path to a directory of images without background.'
    )
    parser.add_argument(
        '-dir2', '--data-dir2',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd2_w2",
        help='Path to a directory of images with background.'
    )
    dir1 = parser.parse_args().data_dir1
    dir2 = parser.parse_args().data_dir2

    # Check directory
    if not dir1.is_dir():
        raise ValueError(f"{dir1} is not a valid directory.")
    elif not dir2.is_dir():
        raise ValueError(f"{dir2} is not a valid directory.")

    main(dir1, dir2, k=1) # For development set -> k=1 
