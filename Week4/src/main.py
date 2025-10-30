
import argparse
import numpy as np
from pathlib import Path

from descriptors import compute_descriptors, deserialize_keypoints_list
from image_split import split_image
from shadow_removal import shadow_removal
from scoring import find_top_ids_for_queries
from evaluations.metrics import mean_average_precision
from background_remover import remove_background_morphological_gradient, crop_images
from evaluations.similarity_measures import compute_similarities
from utils.io_utils import read_images, read_pickle, write_pickle
from utils.plots import plot_query_results

from params import BEST_DESCRIPTOR_PARAMS, BEST_NOISE_PARAMS

SCRIPT_DIR = Path(__file__).resolve().parent


### Week 3 methods ###
"""
I do not know if we should use it or not. Migbt be nice to try the find_top_ids_for_queries with
infer inliners and use inliners True, so it returns the num.paintings it expects. Then try
background + find_top_ids_for_queries (remember to set top_n = 1; check logic in scoring.py).
See which ones has better results.
"""
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

    return splited_images, predictions, painting_counts # For each image, a mask


def process_images(images: list[np.ndarray], background: bool = False):
    """
    Applies denoising and/or background removal to a list of images.
    """

    # Copy to avoid unintended side effects
    processed_images = [img.copy() for img in images]
    painting_counts = [1] * len(processed_images)
    
    if background:
        print("- Background removal -")
        splited_images, masks, painting_counts = remove_background(processed_images)
        processed_images = crop_images(splited_images, masks)

        tmp = []
        for processed_img in processed_images:
            tmp.append(shadow_removal(processed_img,7))
        
        processed_images = tmp
    else:
        print("No background removal")

    return processed_images, painting_counts
    

### Main method ###
def main(dir1: Path) -> None:
    # Create outputs dir where pkl files will be saved
    outputs_dir = SCRIPT_DIR / "outputs_test" 
    outputs_dir.mkdir(exist_ok=True)

    # Central output file for this run
    output_log_file = outputs_dir / "test_evaluation_log.txt"

    print("- Applying BBDD descriptors -")
    method = BEST_DESCRIPTOR_PARAMS["method"]

    # Load/Compute bbdd descriptors
    try:
        print("Loading database descriptors and keypoints...")
        desc_path = SCRIPT_DIR / "descriptors" / f"descriptors_{method}.pkl"
        keys_path = SCRIPT_DIR / "keypoints"   / f"keypoints_{method}.pkl"

        desc_bbdd = read_pickle(desc_path)
        keys_bbdd_serial = read_pickle(keys_path)   # <- aquí lees lo serializado (listas de tuplas)
        keys_bbdd = deserialize_keypoints_list(keys_bbdd_serial)  # <- aquí lo pasas a cv2.KeyPoint

        if len(desc_bbdd) != len(keys_bbdd):
            raise EOFError(f"Longitudes no coinciden: desc={len(desc_bbdd)} keys={len(keys_bbdd)}")


    except FileNotFoundError:
        print("Unable to load database descriptors. Computing them...")
        bbdd_images = read_images(SCRIPT_DIR.parent.parent / "BBDD")
        keys_bbdd, desc_bbdd = compute_descriptors(bbdd_images, 
                                                   BEST_DESCRIPTOR_PARAMS["method"], 
                                                   save_pkl=True)

    tasks = [
        { 
            "name": "QST2_NoDenoised_NoBG",
            "images": read_images(dir1)[:10], # will use the full, just for dev
            "background": False
        }
    ]
    
    print("\n--- Starting Pipeline ---")
    with open(output_log_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write(f"Descriptor: {method}\n")
        f.write(f"Noise Params: {BEST_NOISE_PARAMS}\n")
        f.write("="*30 + "\n\n")

        for task in tasks:
            print(f"\nProcessing task: {task['name']}")
            f.write(f"--- Task: {task['name']} ---\n")

            # Process Images 
            processed_images, _ = process_images(
                task["images"], background=task["background"])
            # Compute Query Descriptors
            print(f"Computing descriptors...")
            keys_q, desc_q = compute_descriptors(processed_images, method=BEST_DESCRIPTOR_PARAMS["method"])

            # Rank: this is really slow, we have all activated, can deactivate things...
            results = find_top_ids_for_queries(
                keys_q, desc_q, keys_bbdd, desc_bbdd,
                desc=BEST_DESCRIPTOR_PARAMS["method"], 
                backend="flann",
                use_mutual=True, 
                use_inliers=True,            
                model="homography", ransac_reproj=3.0,
                T_inl=15, T_ratio=0.30, margin=3,
                top_n=2, # For fallback if we do not use use_inliners and infer_from_inliners, return first, second
                infer_from_inliers=True,
                infer_ratio_drop=0.6
            )

            print(results)
            results_path = outputs_dir / f"results_{task['name']}.pkl" 
            write_pickle(results, results_path)

    print(f"\n--- Pipeline Finished. Full log saved to {output_log_file} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dir1', '--data-dir1',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd1_w4",
        help='Path to a directory of images without background.'
    )
    
    dir1 = parser.parse_args().data_dir1

    # Check directory
    if not dir1.is_dir():
        raise ValueError(f"{dir1} is not a valid directory.")

    main(dir1)