import os
from typing import List

import numpy as np
from pathlib import Path
import cv2

from evaluations.metrics import mean_average_precision
from utils.io_utils import read_pickle
from evaluations.similarity_measures import compute_similarities

from utils.color_spaces import rgb_to_hsv
from utils.histogram import histogram
from utils.io_utils import write_pickle, read_images

SCRIPT_DIR = Path(__file__).resolve().parent

def _spatial_crop(img: np.ndarray, n_crops: int = 3, ) -> np.ndarray:
    """
    Crop an image into n_crops x n_crops regions.
    """
    crops = []
    h, w = img.shape[:2]

    crop_h, crop_w = h // n_crops, w // n_crops
    for i in range(n_crops):
        for j in range(n_crops):
            crops.append(img[i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w])
    return crops

# Helpers
def _dct(img: np.ndarray, keep_coef: int = 8, ) -> np.ndarray:
    """
    Compute the Local Binary Pattern (LBP) representation of a greyscale image. We ignore the border pixels.
    We implemented our own version, but we can also use the one from skimage as their code is optimized in C.

    """

    m = int(np.sqrt(keep_coef))
    dct_img = cv2.dct(img.astype(np.float32))

    values = dct_img[:m, :m].flatten()

    values_norm = values / np.linalg.norm(values)



    return values_norm.astype(np.float32)





def compute_DCT_descriptors(
    imgs: List[np.ndarray], 
    method: str = "dct",
    n_coefs: int = 8,
    greyscale: bool = True,
    n_crops: int = 12,
    save_pkl: bool = False,
    remove_border: bool = False,
    median_filter: bool = False,
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
    
    if median_filter:
        imgs = [cv2.medianBlur(img, 5) for img in imgs]


    if remove_border:
        tmp = []
        for img in imgs:
             h,w = img.shape[:2]
             x = int(0.15*w)
             y = int(0.15*h)
             tmp.append(img[y:h-y,x:w-x])
        imgs = tmp
         
    cropped_imgs = [_spatial_crop(im, n_crops) for im in imgs]
    descs = []

    

    if method == "dct-grayscale" or method == "dct":
        if greyscale:
        
            imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
        cropped_imgs = [_spatial_crop(im, n_crops) for im in imgs]
        initial_descs = [[_dct(crop, keep_coef=n_coefs) for crop in cropped_img] for cropped_img in cropped_imgs]

    elif method == "dct-hsv":
        cropped_imgs = [[rgb_to_hsv(crop) for crop in cropped_img] for cropped_img in cropped_imgs]
        initial_descs = [[np.concatenate([_dct(crop[:,:,i], keep_coef=n_coefs) for i in range(3)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs]
    elif method == "dct-sv":
        cropped_imgs = [[rgb_to_hsv(crop) for crop in cropped_img] for cropped_img in cropped_imgs]
        initial_descs = [[np.concatenate([_dct(crop[:,:,i+1], keep_coef=n_coefs) for i in range(2)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs]
    elif method == "dct-rgb":
        initial_descs = [[np.concatenate([_dct(crop[:,:,i], keep_coef=n_coefs) for i in range(3)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs]
    elif method == "dct-xyz":
        cropped_imgs = [[cv2.cvtColor(crop, cv2.COLOR_BGR2XYZ) for crop in cropped_img] for cropped_img in cropped_imgs]
        initial_descs = [[np.concatenate([_dct(crop[:,:,i], keep_coef=n_coefs) for i in range(3)], axis=0) for crop in cropped_img] for cropped_img in cropped_imgs]
    else:
        raise ValueError(f"Invalid method ({method}) for computing image descriptors!")

    
    
    for cropped_img in initial_descs:

            # Concatenate and normalize
            final_histogram = np.concatenate(cropped_img, axis=0) / (n_crops*n_crops) 
            #TODO: This normalization lacks to be of unit length (it is just the average)
            # final_histogram = final_histogram / np.linalg.norm(final_histogram)  MIGHT NEED TO ADD THIS HERE
            descs.append(final_histogram)

    # Save descriptors to a pickle file
    if save_pkl:
        # Make directory if not setted up
        print("Saving descriptors...")
        os.makedirs(SCRIPT_DIR / "descriptors", exist_ok=True)

        write_pickle(descs, SCRIPT_DIR / "descriptors" / f"{method}.pkl")

    return descs
        
    
# Compute descriptors for benchmark
if __name__=="__main__":
    print("Reading BBDD images...")
    bbdd_imgs = read_images(SCRIPT_DIR.parent.parent / "BBDD")
    query_imgs = read_images(SCRIPT_DIR.parent / "qsd1_w1")
    gt = read_pickle(SCRIPT_DIR.parent / "qsd1_w1" / "gt_corresps.pkl")

    
    grid_search = True   
    scores = {}
    if grid_search:
        
        for n_crop in range(4,5):
            for coef in  [80,90,100,110,120]:
                for method in ["dct-rgb","dct-xyz"]:
                    data_descriptor = compute_DCT_descriptors(bbdd_imgs,n_crops=n_crop,n_coefs=coef, method=method,  save_pkl=False)
                    query_descriptor = compute_DCT_descriptors(query_imgs, n_crops=n_crop,n_coefs=coef, method=method,  save_pkl=False)

                    similarities = compute_similarities(query_descriptor, data_descriptor, metric="euclidean")
                        
                    # Sort the similarities and obtain their indices
                    indices = np.argsort(similarities, axis=1)
                    sorted_sims = np.take_along_axis(similarities, indices, axis=1)

                    # Extract the best k results
                    results_indices = indices[:, :5]
                    results_similarities = sorted_sims[:, :5]
                    map_score = mean_average_precision(indices, gt, k=5)

                    print(f"MAP@{5} score: {map_score:.4f}, using {n_crop} crops, {coef} coefficients.")
                    with open(SCRIPT_DIR / "outputs" / f"{method}_map_scores.txt", "a") as f:
                                f.write(f"MAP@{5} score: {map_score:.4f}, using {method}, {n_crop} crops, {coef} coefficient.\n")
                    
                    results_indices = indices[:, :1]
                    results_similarities = sorted_sims[:, :1]        
                    map_score = mean_average_precision(indices, gt, k=1)

                    print(f"MAP@{1} score: {map_score:.4f}, using {n_crop} crops, {coef} coefficients.")
                    with open(SCRIPT_DIR / "outputs" / f"{method}_map_scores.txt", "a") as f:
                                f.write(f"MAP@{1} score: {map_score:.4f}, using {method}, {n_crop} crops, {coef} coefficient.\n")
