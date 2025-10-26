import argparse
import numpy as np
import cv2
from pathlib import Path
from Week3.src.evaluations.segmentation_eval import evaluation
from background_remover import remove_background_morphological_gradient
from utils.io_utils import read_images
from image_split import split_image
from filter_noise import denoise_batch
from params import best_noise_params

SCRIPT_DIR = Path(__file__).resolve().parent


def get_predictions(images: list[np.ndarray], denoise: bool, out_dir: Path):

    predictions = []

    if denoise:
        images = denoise_batch(images, thresholds=best_noise_params)

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
        
        mask_uint8 = (mask_final * 255).astype("uint8")
        cv2.imwrite(str(out_dir / f"{i:05d}.png"), mask_uint8)

        predictions.append(mask_final)

    return predictions


def main(data_dir: Path) -> None:

    # Read query images to remove background
    og_images = read_images(data_dir)

    # Read ground truths (extension is .png)
    ground_truths = read_images(data_dir, extension="png")

    # Create output directory
    base_out_dir = SCRIPT_DIR / "segmentation_outputs"
    no_denoise_out_dir = base_out_dir / "no_denoising"
    denoise_out_dir = base_out_dir / "denoised"

    no_denoise_out_dir.mkdir(exist_ok=True, parents=True)
    denoise_out_dir.mkdir(exist_ok=True, parents=True)
    
    
    with open(base_out_dir / "evaluation_results.txt", 'w') as f:
        print("Init of evaluations...")

        # Evaluation without denoising images
        predictions_no_denoise = get_predictions(og_images, denoise=False, out_dir=no_denoise_out_dir)
        mean_prec, mean_rec, mean_f1 = evaluation(predictions_no_denoise, ground_truths)

        results_no_denoise = [
            "- Results for background removal WITHOUT denoising images -",
            f"  Mean Precision: {mean_prec:.4f}",
            f"  Mean Recall:    {mean_rec:.4f}",
            f"  Mean F1 Score:  {mean_f1:.4f}",
            "\n" 
        ]
        
        # Print to console and write to file
        for line in results_no_denoise:
            print(line)
            f.write(line + "\n")

        
        # Evaluation of denoising images
        predictions_denoise = get_predictions(og_images, denoise=True, out_dir=denoise_out_dir)
        mean_prec, mean_rec, mean_f1 = evaluation(predictions_denoise, ground_truths)

        results_denoise = [
            "- Results for background removal WITH denoising images -",
            f"  Mean Precision: {mean_prec:.4f}",
            f"  Mean Recall:    {mean_rec:.4f}",
            f"  Mean F1 Score:  {mean_f1:.4f}",
            "\n"
        ]

        # Print to console and write to file
        for line in results_denoise:
            print(line)
            f.write(line + "\n")

    print("\nProcessing complete.")
    

if __name__ == "__main__":

    # Define parsers for data paths
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd2_w3",
        help='Path to the dataset directory.'
    )
    data_dir = parser.parse_args().data_dir
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory.")

    main(data_dir)