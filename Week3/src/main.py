"""
For development set qsd1_w3
"""

import cv2
import argparse
import numpy as np
from pathlib import Path

from dct_descriptors import compute_DCT_descriptors
from background_remover import remove_background_morphological_gradient, crop_images
from image_split import split_image
from evaluations.metrics import mean_average_precision
from evaluations.similarity_measures import compute_similarities
from filter_noise import denoise_batch, plot_image_comparison
from utils.io_utils import read_images, read_pickle, write_pickle
from params import best_desc_params_dct, best_noise_params


SCRIPT_DIR = Path(__file__).resolve().parent
BEST_THRESHOLDS = best_noise_params
BEST_DESC = best_desc_params_dct


def remove_background(images: list[np.ndarray]):
    """
    Removes the background from a list of images.
    Returns a list of boolean masks.
    """
    predictions = []
    
    # Remove background of images
    for i, image in enumerate(images):
        parts = split_image(image) 
        masks = []
        for part in parts:
            _, pred_mask, _, _ = remove_background_morphological_gradient(part)
            masks.append(pred_mask.astype(bool))

        # Combine results of all components
        if len(masks) == 1:
            mask_final = masks[0] 
        else:   
            mask_final = np.concatenate(masks, axis=1) 

        predictions.append(mask_final)

    return predictions


def process_images(images: list[np.ndarray], denoise: bool = False, background: bool = False):
    """
    Applies denoising and/or background removal to a list of images.
    """

    # Copy to avoid unintended side effects
    processed_images = [img.copy() for img in images]
    
    if denoise:
        print("- Denoising images -")
        print("Noise removal parameters: ", BEST_THRESHOLDS)
        processed_images = denoise_batch(processed_images, thresholds=BEST_THRESHOLDS)
        # plot_image_comparison(images1, non_noisy_img1, 5) # plot 5 comparison images
    else:
        print("No denoising of images")

    if background:
        print("- Background removal -")
        masks = remove_background(processed_images)
        processed_images = crop_images(processed_images, masks)
    else:
        print("No background removal")

    return processed_images
    


def main(dir1: Path, dir2: Path, k: int = 10) -> None:

    # Create outputs dir where pkl files will be saved
    outputs_dir = SCRIPT_DIR / "outputs" 
    outputs_dir.mkdir(exist_ok=True)

    # Central output file for this run
    output_log_file = outputs_dir / "tasca4_evaluation_log.txt"

    print("- Applying BBDD descriptors -")
    print("Descriptor parameters:", BEST_DESC)

    n_crops = BEST_DESC["n_crops"]
    n_coefs = BEST_DESC["n_coefs"]
    method = BEST_DESC["method"]

    # Load/Compute bbdd descriptors
    try:
        print("Loading database descriptors...")
        bbdd_descriptors = read_pickle(SCRIPT_DIR / "descriptors" / f"{method}_{n_crops}_{n_coefs}.pkl") # Load descriptors from correct path
    except FileNotFoundError:
        print("Unable to load database descriptors. Computing them...")
        bbdd_images = read_images(SCRIPT_DIR.parent.parent / "BBDD")
        bbdd_descriptors = compute_DCT_descriptors(bbdd_images, n_crops=n_crops, n_coefs=n_coefs, method=method, save_pkl=True) # Add correct path
    
    # Group tasks by dataset for clarity
    tasks = [
        {
            "name": "QSD1_NoDenoise_NoBG",
            "images": read_images(dir1),
            "denoise": False,
            "background": False,
            "gt": read_pickle(dir1 / "gt_corresps.pkl")
        },
        {
            "name": "QSD1_Denoised_NoBG",
            "images": read_images(dir1), # Read again to get a fresh copy
            "denoise": True,
            "background": False,
            "gt": read_pickle(dir1 / "gt_corresps.pkl")
        },
        {
            "name": "QSD2_NoDenoise_BGRemoved",
            "images": read_images(dir2), 
            "denoise": False,
            "background": True,
            "gt": read_pickle(dir2 / "gt_corresps.pkl")
        },
        {
            "name": "QSD2_Denoised_BGRemoved",
            "images": read_images(dir2), # Read again to get a fresh copy
            "denoise": True,
            "background": True,
            "gt": read_pickle(dir2 / "gt_corresps.pkl")
        }
    ]
    
    print("\n--- Starting Pipeline ---")
    with open(output_log_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write(f"Descriptor: {method}, Crops: {n_crops}, Coefs: {n_coefs}\n")
        f.write(f"Noise Params: {BEST_THRESHOLDS}\n")
        f.write("="*30 + "\n\n")

        for task in tasks:
            print(f"\nProcessing task: {task['name']}")
            f.write(f"--- Task: {task['name']} ---\n")

            # Process Images 
            processed_images = process_images(task["images"], denoise=task["denoise"], background=task["background"])

            # Compute Query Descriptors
            print(f"Computing descriptors for {task['name']}...")
            query_descriptors = compute_DCT_descriptors(processed_images, n_crops=n_crops, n_coefs=n_coefs, method=method)

            # Compute Similarities
            similarities = compute_similarities(query_descriptors, bbdd_descriptors, metric="euclidean") # Euclidean (?) TEMPORAL

            # Evaluate
            indices = np.argsort(similarities, axis=1)
            
            map1 = mean_average_precision(indices, task["gt"], k=1)
            map5 = mean_average_precision(indices, task["gt"], k=5)

            # 4.5. Log & Save Results
            print(f"  MAP@1: {map1:.4f}")
            print(f"  MAP@5: {map5:.4f}")
            f.write(f"  MAP@1: {map1:.4f}\n")
            f.write(f"  MAP@5: {map5:.4f}\n\n")

            # Save top k results
            # results = indices[:, :k_results].astype(int).tolist()
            # results_path = outputs_dir / f"results_{task['name']}.pkl" 
            # write_pickle(results, results_path)
            # print(f"  Saved top {k_results} results to {results_path}")

    print(f"\n--- Pipeline Finished. Full log saved to {output_log_file} ---")

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