
import argparse
import numpy as np
from pathlib import Path

from descriptors import sift_descriptors
from evaluations.metrics import mean_average_precision
from evaluations.similarity_measures import compute_similarities
from utils.io_utils import read_images, read_pickle, write_pickle
from utils.plots import plot_query_results



SCRIPT_DIR = Path(__file__).resolve().parent
BEST_THRESHOLDS = best_noise_params
BEST_DESC = best_desc_params_dct



def main(dir1: Path, dir2: Path, k_results: int = 10) -> None:

    # Create outputs dir where pkl files will be saved
    outputs_dir = SCRIPT_DIR / "outputs_test" 
    outputs_dir.mkdir(exist_ok=True)

    # Central output file for this run
    output_log_file = outputs_dir / "test_evaluation_log.txt"

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
            "name": "QST1_Denoised_NoBG",
            "images": read_images(dir1), # Read again to get a fresh copy
            "denoise": True,
            "background": False
        },
        {
            "name": "QST2_Denoised_BGRemoved",
            "images": read_images(dir2), # Read again to get a fresh copy
            "denoise": True,
            "background": True
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

            
            # Compute Query Descriptors
            print(f"Computing descriptors for {task['name']}...")
            query_descriptors = compute_DCT_descriptors(processed_images, n_crops=n_crops, n_coefs=n_coefs, method=method)

            # Compute Similarities
            similarities = compute_similarities(query_descriptors, bbdd_descriptors, metric="euclidean")

            # Evaluate
            indices = np.argsort(similarities, axis=1)

            results = []
            painting_counter = 0
            for i in range(len(task["images"])):
                num_paintings_in_query = painting_counts[i]
                image_indices = indices[painting_counter : painting_counter + num_paintings_in_query]

                if num_paintings_in_query == 1:
                    results.append(image_indices[:, :k_results].tolist())
                else:
                    # Append a list of lists
                    results.append(image_indices[:, :k_results].tolist())
                
                painting_counter += num_paintings_in_query
                

            results_path = outputs_dir / f"results_{task['name']}.pkl" 
            write_pickle(results, results_path)
            print(f"  Saved top {k_results} results to {results_path}")

    print(f"\n--- Pipeline Finished. Full log saved to {output_log_file} ---")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dir1', '--data-dir1',
        type=Path,
        default=SCRIPT_DIR.parent / "qst1_w3",
        help='Path to a directory of images without background.'
    )
    parser.add_argument(
        '-dir2', '--data-dir2',
        type=Path,
        default=SCRIPT_DIR.parent / "qst2_w3",
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