"""
For development set qsd1_w3
"""

import argparse
import numpy as np
from pathlib import Path

from Week3.src.shadow_removal import shadow_removal
from dct_descriptors import compute_DCT_descriptors
from background_remover import remove_background_morphological_gradient, crop_images
from image_split import split_image
from evaluations.metrics import mean_average_precision
from evaluations.similarity_measures import compute_similarities
from filter_noise import denoise_batch
from utils.io_utils import read_images, read_pickle, write_pickle
from utils.plots import plot_query_results
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
    splited_images = []
    painting_counts = []
    
    # Remove background of images
    for i, image in enumerate(images):
        parts = split_image(image)
        # masks = []

        if len(parts) == 2:
            painting_counts.append(2)
        else:
            painting_counts.append(1)
        
        for part in parts:
            _, pred_mask, _, _ = remove_background_morphological_gradient(part)
            splited_images.append(part)
            predictions.append(pred_mask.astype(bool))

    # predictions.append(masks) 
    

    return splited_images, predictions, painting_counts # For each image, a mask


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
    else:
        print("No denoising of images")

    if background:
        print("- Background removal -")
        splited_images, masks, painting_counts = remove_background(processed_images)
        processed_images = crop_images(splited_images, masks)
        tmp = []
        for processed_img in processed_images:
            tmp.append(shadow_removal(processed_img,25))
        
        return processed_images, painting_counts
    else:
        print("No background removal")
        return processed_images, [1] * len(processed_images)
    

def main(dir1: Path, dir2: Path, k_results: int = 10) -> None:

    # Create outputs dir where pkl files will be saved
    outputs_dir = SCRIPT_DIR / "outputs" 
    outputs_dir.mkdir(exist_ok=True)

    # Central output file for this run
    output_log_file = outputs_dir / "descriptors_evaluation_log.txt"

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
    

    tasks = [
        
        
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
            processed_images, painting_counts = process_images(task["images"], denoise=task["denoise"], background=task["background"]) 

            processed_images

            # Compute Query Descriptors
            print(f"Computing descriptors for {task['name']}...")
            query_descriptors = compute_DCT_descriptors(processed_images, n_crops=n_crops, n_coefs=n_coefs, method=method)

            # Compute Similarities
            similarities = compute_similarities(query_descriptors, bbdd_descriptors, metric="euclidean")

            # Evaluate
            indices = np.argsort(similarities, axis=1)
            sorted_sims = np.take_along_axis(similarities, indices, axis=1)

            # In case of two paintings in one image
            gt = [[item] for sublist in task["gt"] for item in sublist]

            map1 = mean_average_precision(indices, gt, k=1)
            map5 = mean_average_precision(indices, gt, k=5)

            # 4.5. Log & Save Results
            print(f"  MAP@1: {map1:.4f}")
            print(f"  MAP@5: {map5:.4f}")
            f.write(f"  MAP@1: {map1:.4f}\n")
            f.write(f"  MAP@5: {map5:.4f}\n\n")

            plot_query_results(processed_images, indices[:, :5], sorted_sims[:, :5], k=5,  
                                    save_path=Path(__file__).resolve().parent / 'outputs' / f'query_at{5}_{task["name"]}.png')

            
            # # THIS PART OF THE CODE WILL BE USEFUL JUST FOR THE TEST SETS
            # results = []
            # painting_counter = 0
            # for i in range(len(task["images"])):
            #     num_paintings_in_query = painting_counts[i]
            #     image_indices = indices[painting_counter : painting_counter + num_paintings_in_query]

            #     if num_paintings_in_query == 1:
            #         results.append(image_indices[:, :k_results].tolist())
            #     else:
            #         # Append a list of lists
            #         results.append(image_indices[:, :k_results].tolist())
                
            #     painting_counter += num_paintings_in_query
                

            # results_path = outputs_dir / f"results_{task['name']}.pkl" 
            # write_pickle(results, results_path)
            # print(f"  Saved top {k_results} results to {results_path}")

    print(f"\n--- Pipeline Finished. Full log saved to {output_log_file} ---")



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