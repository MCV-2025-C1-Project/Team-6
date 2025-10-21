import os
import cv2
import numpy as np
import argparse
from typing import List
from pathlib import Path
import pywt  # <-- Added for Wavelet Transform

from evaluations.metrics import mean_average_precision
from utils.io_utils import read_pickle
from evaluations.similarity_measures import compute_similarities
from utils.color_spaces import rgb_to_hsv
from utils.io_utils import write_pickle, read_images
# from params import best_desc_params_dct, dct_search_space # Original
from Week3.src.shadow_removal import shadow_removal

from background_remover import remove_background_morphological_gradient, crop_images
from image_split import split_image
from evaluations.metrics import mean_average_precision
from evaluations.similarity_measures import compute_similarities
from filter_noise import denoise_batch
from utils.io_utils import read_images, read_pickle, write_pickle
from utils.plots import plot_query_results
# from params import best_desc_params_dct, best_noise_params # Original

# --- Wavelet Substitution ---
# Define placeholder params for wavelet search
best_desc_params_wavelet = {
    'n_crops': 12,
    'level': 3,
    'wavelet_type': 'haar',
    'method': 'wavelet-hsv'
}
wavelet_search_space = {
    'n_crops': [2, 3, 4],
    'level': [2, 3, 4],
    'wavelet_type': ['haar', 'db1', 'db2'],
    'method': ['wavelet-grayscale', 'wavelet-hsv', 'wavelet-rgb', 'wavelet-sv']
}
# Placeholder for noise params, kept from original logic
best_noise_params = {} 
# --- End Wavelet Substitution ---


SCRIPT_DIR = Path(__file__).resolve().parent
# BEST_PARAMS = best_desc_params_dct # Original
# SEARCH_SPACE = dct_search_space # Original
BEST_PARAMS = best_desc_params_wavelet
SEARCH_SPACE = wavelet_search_space
BEST_THRESHOLDS = best_noise_params
# BEST_DESC = best_desc_params_dct # Original
BEST_DESC = best_desc_params_wavelet


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
            tmp.append(shadow_removal(processed_img,45))
        
        return tmp, painting_counts
    else:
        print("No background removal")
        return processed_images, [1] * len(processed_images)
    

def _spatial_crop(img: np.ndarray, n_crops: int = 3) -> List[np.ndarray]:
    """
    Crop an image into n_crops x n_crops regions.

    Args:
        img: The input image as a numpy array (H, W, C) or (H, W).
        n_crops: The number of crops per dimension (e.g., 3 -> 3x3 grid = 9 crops).
    Returns:
        A list of numpy arrays, where each element is a sub-image (crop).
    """
    crops = []
    # Get image height and width
    h, w = img.shape[:2]

    # Calculate the dimensions of each individual crop using integer division
    crop_h, crop_w = h // n_crops, w // n_crops

    for i in range(n_crops):
        for j in range(n_crops):
            crop = img[i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w]
            crops.append(crop)

    return crops


# --- Wavelet Substitution: Replaced _dct with _wavelet ---
# Helpers
def _wavelet(img: np.ndarray, level: int = 2, wavelet_type: str = 'haar') -> np.ndarray:
    """
    Compute Wavelet approximation coefficients for a single-channel image.
    This function replaces the original _dct function.
    
    Args:
        img: Input image (or patch), expected to be single-channel.
        level: The decomposition level for the wavelet transform.
        wavelet_type: The type of wavelet to use (e.g., 'haar', 'db1').
    Returns:
        A 1D L2-normalized feature vector of the approximation coefficients.
    """
    # Ensure input is float
    img_float = img.astype(np.float32)

    # Handle potential empty or too small images for the given level
    min_size = 2**level
    if img_float.shape[0] < min_size or img_float.shape[1] < min_size:
        # If too small, resize it to be just large enough
        # This is a fallback, ideally crops are large enough
        img_float = cv2.resize(img_float, (min_size, min_size), interpolation=cv2.INTER_AREA)

    # Compute the 2D Wavelet Decomposition
    # Use 'reflect' mode for border handling
    coeffs = pywt.wavedec2(img_float, wavelet_type, level=level, mode='reflect')
    
    # Extract the top-level approximation coefficients (low-frequency)
    approx_coeffs = coeffs[0]
    
    # Flatten into a 1D vector
    values = approx_coeffs.flatten()

    # L2-normalize the vector
    norm = np.linalg.norm(values)
    
    # Avoid division by zero
    if norm == 0:
        return values.astype(np.float32)  # Return the zero vector
        
    values_norm = values / norm

    return values_norm.astype(np.float32)
# --- End Wavelet Substitution ---


# --- Wavelet Substitution: Replaced compute_DCT_descriptors with compute_Wavelet_descriptors ---
def compute_Wavelet_descriptors(
    imgs: List[np.ndarray], 
    method: str = "wavelet",
    level: int = 2,
    wavelet_type: str = 'haar',
    greyscale: bool = True, # This param seems unused in original logic, but kept for style
    n_crops: int = 12,
    save_pkl: bool = False,
    remove_border: bool = False,
) -> List[np.ndarray]:
    """
    Compute spatial wavelet descriptors for a list of images.
    This function replaces compute_DCT_descriptors, following its style.

    Args:
        imgs: List of images as numpy arrays.
        method: Color space/method ("wavelet", "wavelet-grayscale", "wavelet-hsv", "wavelet-sv", "wavelet-rgb", "wavelet-xyz").
        level: Decomposition level for the wavelet transform.
        wavelet_type: Type of wavelet to use (e.g., 'haar').
        n_crops: Number of crops per dimension (n_crops x n_crops).
        save_pkl: Whether to save the computed descriptors as a pickle file.
        remove_border: Whether to crop image borders.

    Returns:
        List of descriptors as numpy arrays.
    """

    # Remove image borders
    if remove_border:
        tmp = []
        for img in imgs:
             h,w = img.shape[:2]
             x = int(0.15*w)
             y = int(0.15*h)
             tmp.append(img[y:h-y,x:w-x])
        imgs = tmp

    # Divide all images into crops.         
    cropped_imgs = [_spatial_crop(im, n_crops) for im in imgs]
    
    descs = []
    initial_descs = []
    
    # Helper function to apply wavelet
    def get_wavelet_desc(crop, **kwargs):
        return _wavelet(crop, level=level, wavelet_type=wavelet_type, **kwargs)

    # Compute descriptors based on the chosen method
    if method == "wavelet-grayscale" or method == "wavelet":
        # This branch handles greyscale Wavelet
        imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
        cropped_imgs_gray = [_spatial_crop(im, n_crops) for im in imgs_gray]
        initial_descs = [[get_wavelet_desc(crop) for crop in cropped_img] for cropped_img in cropped_imgs_gray]
    elif method == "wavelet-hsv":
        # Convert each crop to HSV
        cropped_imgs_hsv = [[rgb_to_hsv(crop) for crop in cropped_img] for cropped_img in cropped_imgs]
        initial_descs = [[np.concatenate([get_wavelet_desc(crop[:,:,i]) for i in range(3)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs_hsv]
    elif method == "wavelet-sv":
        # Convert each crop to HSV
        cropped_imgs_hsv = [[rgb_to_hsv(crop) for crop in cropped_img] for cropped_img in cropped_imgs]
        initial_descs = [[np.concatenate([get_wavelet_desc(crop[:,:,i+1]) for i in range(2)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs_hsv]
    elif method == "wavelet-rgb":
        # No color conversion needed
        initial_descs = [[np.concatenate([get_wavelet_desc(crop[:,:,i]) for i in range(3)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs]
    elif method == "wavelet-xyz":
        # Convert each crop to XYZ color space
        cropped_imgs_xyz = [[cv2.cvtColor(crop, cv2.COLOR_BGR2XYZ) for crop in cropped_img] for cropped_img in cropped_imgs]
        initial_descs = [[np.concatenate([get_wavelet_desc(crop[:,:,i]) for i in range(3)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs_xyz]
    else:
        raise ValueError(f"Invalid method ({method}) for computing image descriptors!")

    # Aggregate patch descriptors for each image
    for cropped_img in initial_descs:
            # Concatenate and normalize (following original style)
            final_histogram = np.concatenate(cropped_img, axis=0) / (n_crops*n_crops) 
            #TODO: This normalization lacks to be of unit length (it is just the average)
            # final_histogram = final_histogram / np.linalg.norm(final_histogram)  MIGHT NEED TO ADD THIS HERE
            descs.append(final_histogram)

    # Save descriptors to a pickle file
    if save_pkl:
        # Make directory if not setted up
        os.makedirs(SCRIPT_DIR / "descriptors", exist_ok=True)
        
        print("Saving descriptors...")
        write_pickle(descs, SCRIPT_DIR / "descriptors" / f"{method}_{n_crops}_{level}_{wavelet_type}.pkl")

    return descs
# --- End Wavelet Substitution ---
        
    
# Compute descriptors for benchmark
if __name__=="__main__":

    # Define parsers for data paths
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dir', '--data-dir',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd2_w3",
        help='Path to a directory of images with noise.'
    )
    parser.add_argument(
        '-bd', '--bbdd-dir',
        type=Path,
        default=SCRIPT_DIR.parent.parent / "BBDD",
        help='Path to the database directory.'
    )
    parser.add_argument(
        '-gs', '--grid-search',
        type=bool,
        default=False,
        help='Parameter that controls whether to perform grid search.'
    )
    dir = parser.parse_args().data_dir
    bd_dir = parser.parse_args().bbdd_dir
    grid_search = parser.parse_args().grid_search
    if not dir.is_dir():
        raise ValueError(f"{dir} is not a valid directory.")
    if not bd_dir.is_dir():
        raise ValueError(f"{bd_dir} is not a valid directory.")
    
    # Make sure output directory exists
    (SCRIPT_DIR / "outputs").mkdir(exist_ok=True)

    print("Reading BBDD images...")
    bbdd_imgs = read_images(bd_dir)
    
    scores = {}
    if grid_search:
        print("Init of Grid Search (Wavelet)...")
        query_imgs = read_images(dir)
        gt = read_pickle(dir / "gt_corresps.pkl")
        processed_images, painting_counts = process_images(query_imgs, denoise=True, background=True)
        
        # --- Wavelet Substitution ---
        for n_crop in SEARCH_SPACE['n_crops']:
            for lvl in SEARCH_SPACE['level']:
                for wt in SEARCH_SPACE['wavelet_type']:
                    for method in SEARCH_SPACE['method']:
                        data_descriptor = compute_Wavelet_descriptors(bbdd_imgs,
                                                                    n_crops=n_crop,
                                                                    level=lvl,
                                                                    wavelet_type=wt,
                                                                    method=method, 
                                                                    save_pkl=False)
                        query_descriptor = compute_Wavelet_descriptors(processed_images, 
                                                                     n_crops=n_crop,
                                                                     level=lvl, 
                                                                     wavelet_type=wt,
                                                                     method=method, 
                                                                     save_pkl=False)
                        
                        similarities = compute_similarities(query_descriptor, data_descriptor, metric="euclidean")
                            
                        # Sort the similarities and obtain their indices
                        indices = np.argsort(similarities, axis=1)
                        sorted_sims = np.take_along_axis(similarities, indices, axis=1)

                        gtt = [[item] for sublist in gt for item in sublist]  
                        gt = gtt

                        # Extract the best k results
                        results_indices = indices[:, :5]
                        results_similarities = sorted_sims[:, :5]
                        map_score = mean_average_precision(indices, gt, k=5)

                        # Print and save scores por MAP@5
                        print(f"MAP@{5} score: {map_score:.4f}, using {n_crop} crops, level {lvl}, wavelet {wt}, method {method}.")
                        with open(SCRIPT_DIR / "outputs" / f"{method}_map_scores.txt", "a") as f:
                            f.write(f"MAP@{5} score: {map_score:.4f}, using {method}, {n_crop} crops, level {lvl}, wavelet {wt}.\n")
                        
                        results_indices = indices[:, :1]
                        results_similarities = sorted_sims[:, :1]        
                        map_score = mean_average_precision(indices, gt, k=1)

                        # Print and save scores por MAP@1
                        print(f"MAP@{1} score: {map_score:.4f}, using {n_crop} crops, level {lvl}, wavelet {wt}, method {method}.")
                        with open(SCRIPT_DIR / "outputs" / f"{method}_map_scores.txt", "a") as f:
                            f.write(f"MAP@{1} score: {map_score:.4f}, using {method}, {n_crop} crops, level {lvl}, wavelet {wt}.\n")
        # --- End Wavelet Substitution ---
    else:
        # Compute descriptor with the best parameters
        # --- Wavelet Substitution ---
        print(f"Computing best Wavelet descriptors with params: {BEST_PARAMS}")
        data_descriptor = compute_Wavelet_descriptors(bbdd_imgs, 
                                                n_crops=BEST_PARAMS['n_crops'], 
                                                level=BEST_PARAMS['level'],
                                                wavelet_type=BEST_PARAMS['wavelet_type'],
                                                method=BEST_PARAMS['method'], 
                                                save_pkl=True)
        print("Descriptors computed and saved.")
        # --- End Wavelet Substitution ---