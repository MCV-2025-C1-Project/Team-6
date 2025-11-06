import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
import cv2 as cv
from descriptors import compute_descriptors, deserialize_keypoints_list
from image_split import split_image
from shadow_removal import shadow_removal
from scoring import find_top_ids_for_queries
from background_remover import remove_background_morphological_gradient, crop_images
from utils.io_utils import read_images, read_pickle, write_pickle

from params import BEST_DESCRIPTOR_PARAMS

SCRIPT_DIR = Path(__file__).resolve().parent


# Helper
def to_py_int_results(results):
    cleaned = []
    for row in results:
        if isinstance(row, np.ndarray):
            row = row.tolist()
        cleaned.append([int(x) for x in row])
    return cleaned

### Week 3 methods ###
def split_images(images: list[np.ndarray]) -> Tuple[list[np.ndarray], list[int]]:
    """
    Split images into parts (0/1/2) WITHOUT background removal.
    Returns:
      - split_parts: flat list of all parts in order
      - painting_counts: per original image, number of parts produced (1 or 2)
    """
    split_parts: list[np.ndarray] = []
    painting_counts: list[int] = []

    for img in images:
        parts = split_image(img)
        painting_counts.append(2 if len(parts) == 2 else 1)
        split_parts.extend(parts)

    return split_parts, painting_counts

def remove_background(parts: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Run BG removal on already-split parts.
    Returns:
      - cropped_parts: list[np.ndarray] after cropping with masks
      - masks: list[np.ndarray] boolean masks (same length as parts)
    """
    masks: list[np.ndarray] = []
    for part in parts:
        _, pred_mask, _, _ = remove_background_morphological_gradient(part)
        masks.append(pred_mask.astype(bool))

    cropped_parts = crop_images(parts, masks)
    return cropped_parts, masks

def process_images(
    images: list[np.ndarray],
    split: bool = True,
    background: bool = False,
    do_shadow: bool = False,
    shadow_ksize: int = 7) -> tuple[list[np.ndarray], list[int]]:
    """
    1) optionally split
    2) optionally background removal (requires split if True)
    3) optionally shadow removal
    Returns:
      - processed_images (list[np.ndarray])
      - painting_counts (list[int]) aligned to original inputs
    """

    noise_removed = []
    for img in images:
        noise_removed.append( cv.medianBlur(img, 5)  # Kernel size is 5 (must be odd)
    )
    

    if split:
        parts, painting_counts = split_images(noise_removed)
    else:
        parts = [img.copy() for img in images]
        painting_counts = [1] * len(images)

    if background:
        parts, _ = remove_background(parts)

    if do_shadow:
        parts = [shadow_removal(p, shadow_ksize) for p in parts]

    return parts, painting_counts
    

### Main method ###
def main(dir1: Path) -> None:
    outputs_dir = SCRIPT_DIR / "outputs_test"
    outputs_dir.mkdir(exist_ok=True)
    output_log_file = outputs_dir / "test_evaluation_log.txt"

    print("- Applying BBDD descriptors -")
    method = BEST_DESCRIPTOR_PARAMS["method"]

    # Load/Compute bbdd descriptors
    try:
        print("Loading database descriptors and keypoints...")
        desc_path = SCRIPT_DIR / "descriptors" / f"descriptors_{method}.pkl"
        keys_path = SCRIPT_DIR / "keypoints"   / f"keypoints_{method}.pkl"

        desc_bbdd = read_pickle(desc_path)
        keys_bbdd_serial = read_pickle(keys_path)  
        keys_bbdd = deserialize_keypoints_list(keys_bbdd_serial)  

        if len(desc_bbdd) != len(keys_bbdd):
            raise EOFError(f"Length mismatch: desc={len(desc_bbdd)} keys={len(keys_bbdd)}")

    except FileNotFoundError:
        print("Unable to load database descriptors. Computing them...")
        bbdd_images = read_images(SCRIPT_DIR.parent.parent / "BBDD")
        keys_bbdd, desc_bbdd = compute_descriptors(bbdd_images, 
                                                   method=BEST_DESCRIPTOR_PARAMS["method"], 
                                                   save_pkl=True)

    tasks = [
        {
            "name": "QSD4",
            "images": read_images(dir1),
            "split": BEST_DESCRIPTOR_PARAMS["split"],
        }
    ]
    
    print("\n--- Starting Pipeline ---")
    with open(output_log_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write(f"Descriptor: {method}\n")
        f.write("="*30 + "\n\n")

        for task in tasks:
            print(f"\nProcessing task: {task['name']}")
            f.write(f"--- Task: {task['name']} ---\n")

            # Process Images 
            processed_images, painting_counts = process_images(task["images"], split=task["split"], background=False)

            # Compute Query Descriptors
            print(f"Computing descriptors...")
            keys_q, desc_q = compute_descriptors(processed_images, 
                                                method=BEST_DESCRIPTOR_PARAMS["method"],
                                                save_pkl=False)

            # Rank: this is really slow, we have all activated, can deactivate things...
            results = find_top_ids_for_queries(
                queries_kp=keys_q,
                queries_desc=desc_q,
                bbdd_kp=keys_bbdd,
                bbdd_desc=desc_bbdd, 
                paint_counts=painting_counts,       
                desc=method,
                backend=BEST_DESCRIPTOR_PARAMS["backend"],
                use_mutual=True,
                use_inliers=True,
                model="homography",
                ransac_reproj=BEST_DESCRIPTOR_PARAMS["ransac_reproj"],
                # inference / calibrated knobs
                infer_from_inliers=True,
                T_inl=BEST_DESCRIPTOR_PARAMS["T_inl"],
                T_ratio=BEST_DESCRIPTOR_PARAMS["T_ratio"],
                T_peak_ratio=BEST_DESCRIPTOR_PARAMS["T_peak_ratio"],
                T_z=BEST_DESCRIPTOR_PARAMS["T_z"],
                k_stat=BEST_DESCRIPTOR_PARAMS["k_stat"],
                top_n=2,
                # mode
                splits=task["split"]
            )

            results = to_py_int_results(results)
            print(f'RESULTS: {results}')
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