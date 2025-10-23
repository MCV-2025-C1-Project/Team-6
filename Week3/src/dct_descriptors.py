import os

import cv2
import numpy as np
import argparse
from typing import List
from pathlib import Path

from evaluations.metrics import mean_average_precision
from utils.io_utils import read_pickle
from evaluations.similarity_measures import compute_similarities
from utils.color_spaces import rgb_to_hsv
from utils.io_utils import write_pickle, read_images
from params import best_desc_params_dct, dct_search_space
from shadow_removal import shadow_removal

from background_remover import remove_background_morphological_gradient, crop_images
from image_split import split_image
from evaluations.metrics import mean_average_precision
from evaluations.similarity_measures import compute_similarities
from filter_noise import denoise_batch
from utils.io_utils import read_images, read_pickle, write_pickle
from utils.plots import plot_query_results
from params import best_desc_params_dct, best_noise_params

# TODO: explained in compute_dct_descriptors

SCRIPT_DIR = Path(__file__).resolve().parent
BEST_PARAMS = best_desc_params_dct
SEARCH_SPACE = dct_search_space
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

def create_rgb_grayscale(bgr_img):
    """
    Takes a 3-channel BGR image and returns a 4-channel
    image with (R, G, B, Grayscale) channels.
    """
    # 1. Get the R, G, B channels
    # The input 'crop' is BGR, so split gives (b, g, r)
    b, g, r = cv2.split(bgr_img)
    
    # 2. Get the Grayscale channel
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    # 3. Merge them in the order R, G, B, Grayscale
    return cv2.merge((r, g, b, gray))

def process_images(images: list[np.ndarray], denoise: bool = False, background: bool = False,threshold: int = 14):
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
            tmp.append(shadow_removal(processed_img,threshold))
        
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


# Helpers
def _dct(img: np.ndarray, keep_coef: int = 8) -> np.ndarray:
    """
    Compute the Local Binary Pattern (LBP) representation of a greyscale image. We ignore the border pixels.
    We implemented our own version, but we can also use the one from skimage as their code is optimized in C.
    
    Args:
        img: Input image (or patch), expected to be single-channel (greyscale).
        keep_coef: A number used to determine the m x m block of coefficients to keep.
    Returns:
        A 1D L2-normalized feature vector of the low-frequency DCT coefficients.
    """
    # Calculate the side length 'm' of the square block of coefficients to keep.
    m = int(np.sqrt(keep_coef))
    
    # Compute the 2D DCT. Input must be float32 for cv2.dct.
    dct_img = cv2.dct(img.astype(np.float32))

    # Extract the top-left m x m block, which contains the low-frequency components.
    # Then, flatten it into a 1D vector.
    values = dct_img[:m, :m].flatten()

    # L2-normalize the vector to make it robust to illumination changes.
    norm = np.linalg.norm(values)
    
    # Add a safety check to avoid division by zero if the patch is all black
    if norm == 0:
        return values.astype(np.float32)  # Return the zero vector
        
    values_norm = values / norm

    return values_norm.astype(np.float32)


def compute_DCT_descriptors(
    imgs: List[np.ndarray], 
    method: str = "dct",
    n_coefs: int = 8,
    greyscale: bool = True,
    n_crops: int = 12,
    save_pkl: bool = False,
    remove_border: bool = False,
) -> List[np.ndarray]:
    """
    Compute spatial color descriptors for a list of images. 

    Args:
        imgs: List of images as numpy arrays.
        n_crops: Number of crops per dimension (n_crops x n_crops).
        pyramid: Whether to use spatial pyramid representation.
        center_weights: Whether to apply center weights to crops.
        pyramid_levels: List of levels for the spatial pyramid (only if pyramid=True).
        method: Color space/method for descriptor computation ("rgb", "hs", "hsv", "rgb-hs", "rgb-hsv").
        n_bins: Number of bins for the histograms.
        save_pkl: Whether to save the computed descriptors as a pickle file.

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

    # Compute descriptors based on the chosen method
    if method == "dct-grayscale" or method == "dct":
        # This branch handles greyscale DCT
        if greyscale:
            imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
        cropped_imgs = [_spatial_crop(im, n_crops) for im in imgs]
        initial_descs = [[_dct(crop, keep_coef=n_coefs) for crop in cropped_img] for cropped_img in cropped_imgs]
    elif method == "dct-hsv":
        # Convert each crop to HSV
        cropped_imgs = [[rgb_to_hsv(crop) for crop in cropped_img] for cropped_img in cropped_imgs]
        initial_descs = [[np.concatenate([_dct(crop[:,:,i], keep_coef=n_coefs) for i in range(3)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs]
    elif method == "dct-sv":
        # Convert each crop to HSV
        cropped_imgs = [[rgb_to_hsv(crop) for crop in cropped_img] for cropped_img in cropped_imgs]
        initial_descs = [[np.concatenate([_dct(crop[:,:,i+1], keep_coef=n_coefs) for i in range(2)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs]
    elif method == "dct-rgb":
        # No color conversion needed
        initial_descs = [[np.concatenate([_dct(crop[:,:,i], keep_coef=n_coefs) for i in range(3)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs]
    elif method == "dct-xyz":
        # Convert each crop to XYZ color space
        cropped_imgs = [[cv2.cvtColor(crop, cv2.COLOR_BGR2XYZ) for crop in cropped_img] for cropped_img in cropped_imgs]
        initial_descs = [[np.concatenate([_dct(crop[:,:,i], keep_coef=n_coefs) for i in range(3)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs]
    elif method == "dct-rgbg":
        # Convert each crop to XYZ color space
        cropped_imgs = [[create_rgb_grayscale(crop) for crop in cropped_img] for cropped_img in cropped_imgs]
        initial_descs = [[np.concatenate([_dct(crop[:,:,i], keep_coef=n_coefs) for i in range(4)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs]
    else:
        raise ValueError(f"Invalid method ({method}) for computing image descriptors!")

    # Aggregate patch descriptors for each image
    for cropped_img in initial_descs:
            # Concatenate and normalize
            final_histogram = np.concatenate(cropped_img, axis=0) / (n_crops*n_crops) 
            #TODO: This normalization lacks to be of unit length (it is just the average)
            # final_histogram = final_histogram / np.linalg.norm(final_histogram)  MIGHT NEED TO ADD THIS HERE
            descs.append(final_histogram)

    # Save descriptors to a pickle file
    if save_pkl:
        # Make directory if not setted up
        os.makedirs(SCRIPT_DIR / "descriptors", exist_ok=True)
        
        print("Saving descriptors...")
        write_pickle(descs, SCRIPT_DIR / "descriptors" / f"{method}_{n_crops}_{n_coefs}.pkl")

    return descs
        
    
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
        print("Init of Grid Search...")
        query_imgs = read_images(dir)
        gt = read_pickle(dir / "gt_corresps.pkl")
        for threshold in SEARCH_SPACE['thresholds']:
            processed_images, painting_counts = process_images(query_imgs, denoise=True, background=True,threshold = threshold) 
        
            for axises in SEARCH_SPACE['axises']:
                for directions in SEARCH_SPACE['directions']:
                    for n_crop in SEARCH_SPACE['n_crops']:
                        for coef in SEARCH_SPACE['n_coefs']:
                            for method in SEARCH_SPACE['method']:
                                data_descriptor = compute_DCT_descriptors(bbdd_imgs,n_crops=n_crop,n_coefs=coef, method=method, save_pkl=False)
                                query_descriptor = compute_DCT_descriptors(processed_images, n_crops=n_crop,n_coefs=coef, method=method, save_pkl=False)
                                
                                similarities = compute_similarities(query_descriptor, data_descriptor, metric="euclidean")
                                    
                                # Sort the similarities and obtain their indices
                                indices = np.argsort(similarities, axis=1)
                                sorted_sims = np.take_along_axis(similarities, indices, axis=1)
                                gtt = [[item] for sublist in gt for item in sublist]    
                                # Extract the best k results
                                results_indices = indices[:, :5]
                                results_similarities = sorted_sims[:, :5]
                                map_score = mean_average_precision(indices, gtt, k=5)

                                # Print and save scores por MAP@5
                                print(f"MAP@{5} score: {map_score:.4f}, using {n_crop} crops, {coef} coefficients.")
                                with open(SCRIPT_DIR / "outputs" / f"{method}_map_scores.txt", "a") as f:
                                    f.write(f"MAP@{5} score: {map_score:.4f}, using {method}, {n_crop} crops, {coef} coefficient.\n")
                                
                                results_indices = indices[:, :1]
                                results_similarities = sorted_sims[:, :1]  
                                # In case of two paintings in one image
                                
                                map_score = mean_average_precision(indices, gtt, k=1)

                                # Print and save scores por MAP@1
                                print(f"MAP@{1} score: {map_score:.4f}, using {n_crop} crops, {coef} coefficients.")
                                with open(SCRIPT_DIR / "outputs" / f"{method}{axises}{directions}_map_scores.txt", "a") as f:
                                    f.write(f"MAP@{1} score: {map_score:.4f}, using {method}, {n_crop} crops, {coef} coefficient.\n")
    else:
        # Compute descriptor with the best parameters
        data_descriptor = compute_DCT_descriptors(bbdd_imgs, 
                                                n_crops=BEST_PARAMS['n_crops'], 
                                                n_coefs=BEST_PARAMS['n_coefs'], 
                                                method=BEST_PARAMS['method'], 
                                                save_pkl=True)