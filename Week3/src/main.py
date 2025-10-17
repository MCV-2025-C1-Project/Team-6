import argparse
from pathlib import Path

import numpy as np

from dct_descriptors import compute_DCT_descriptors

from background import apply_segmentation
from evaluations.similarity_measures import compute_similarities
from filter_noise import denoise_batch, plot_image_comparison
from utils.io_utils import read_images, read_pickle, write_pickle
from params import best_desc_params, best_noise_params, best_segmentation_params

SCRIPT_DIR = Path(__file__).resolve().parent


def main(dir1: Path, dir2: Path, k: int = 10) -> None:

    # Create outputs dir where pkl files will be saved
    outputs_dir = SCRIPT_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Noise removal params
    noise_params = best_noise_params
    print("Noise removal parameters:", noise_params)

    # Descriptor parameters
    desc_params = best_desc_params
    print("Descriptor parameters:", desc_params)

    n_crops = desc_params["n_crops"]
    n_coefs = desc_params["n_coefs"]
    method = desc_params["method"]

    # Segmentation parameters
    segm_params = best_segmentation_params
    print("Segmentation parameters:", segm_params)

    try:
        print("Loading database descriptors...")
        bbdd_descriptors = read_pickle(SCRIPT_DIR.parent /"descriptors" / f"{method}_{n_crops}_{n_coefs}.pkl") # Load descriptors from correct path
    except FileNotFoundError:
        print("Unable to load database descriptors. Computing them...")
        bbdd_images = read_images(SCRIPT_DIR.parent.parent / "BBDD")
        bbdd_descriptors = compute_DCT_descriptors(bbdd_images,n_crops=n_crops,n_coefs=n_coefs, method=method,  save_pkl=True) # Add correct path

    """Process dataset of images without background."""
    print("Processing dataset of images without background...")

    # Read query images
    images1 = read_images(dir1)

    # Remove noise
    non_noisy_img1 = denoise_batch(images1, thresholds=best_noise_params)
    plot_image_comparison(images1, non_noisy_img1, 5) # plot 5 comparison images

    # Compute query images descriptors

    

    descriptors = compute_DCT_descriptors(non_noisy_img1,n_crops=n_crops,n_coefs=n_coefs, method=method,  save_pkl=False) #TODO: Check how this will work

    # Compute similarities
    similarities = compute_similarities(descriptors, bbdd_descriptors, desc_params["metric"])

    # Sort the indices resulting from the similarities sorting
    indices = np.argsort(similarities, axis=1)

    # Save results for k
    results = indices[:, :k].astype(int).tolist()
    write_pickle(results, outputs_dir / NAME) #TODO: Check

    """Process dataset of images with background."""
    print("Processing dataset of images with background...")

    # Read query images
    images2 = read_images(dir2)

    # Remove noise
    non_noisy_img2 = denoise_batch(images2, thresholds=best_noise_params)
    plot_image_comparison(images2, non_noisy_img2, 5) # plot 5 comparison images

    # Detect BG from images
    masks = apply_segmentation(images2, segm_params, save_plot=True)

    # Crop paintings
    paintings = crop_images(images2, masks) #TODO: do we do this this week?

    # Compute query images descriptors
    descriptors = compute_DCT_descriptors(paintings,n_crops=n_crops,n_coefs=n_coefs, method=method,  save_pkl=False) 

    # Compute similarities
    similarities = compute_similarities(descriptors, bbdd_descriptors, desc_params["metric"])

    # Sort the indices resulting from the similarities sorting
    indices = np.argsort(similarities, axis=1)

    # Save results for k
    results = indices[:, :k].astype(int).tolist()
    write_pickle(results, outputs_dir / NAME) #TODO: Check 

    # TODO: Handle submission of results in correct format


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dir1', '--data-dir1',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd1_w3",
        help='Path to a directory of images without background.'
    )
    parser.add_argument(
        '-dir2', '--data-dir2',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd2_w3",
        help='Path to a directory of images with background.'
    )
    dir1 = parser.parse_args().data_dir1
    dir2 = parser.parse_args().data_dir2

    # Check directory
    if not dir1.is_dir():
        raise ValueError(f"{dir1} is not a valid directory.")
    elif not dir2.is_dir():
        raise ValueError(f"{dir2} is not a valid directory.")

    main(dir1, dir2)