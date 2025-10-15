import os
from typing import List
from skimage import feature
import numpy as np
from pathlib import Path
import cv2
from params import lbp_testing
from utils.color_spaces import rgb_to_hsv
from utils.histogram import histogram
from utils.io_utils import write_pickle, read_images
from evaluations.metrics import mean_average_precision
from utils.io_utils import read_pickle
from evaluations.similarity_measures import compute_similarities
from params import lbp_testing
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
def _simple_lbp(img: np.ndarray,  cv2_implementation: bool = True) -> np.ndarray:
    """
    Compute the Local Binary Pattern (LBP) representation of a greyscale image. We ignore the border pixels.
    We implemented our own version, but we can also use the one from skimage as their code is optimized in C.

    """

    n_bins = 256
    
         
    if cv2_implementation:
        lbp_map = feature.local_binary_pattern(img, P=8, R=1, method="default")
        lbp_map = lbp_map.astype(np.float32) / 255.0
        pLBP = histogram(lbp_map, n_bins=n_bins) 
        return pLBP.astype(np.float32)

    else:
        height, width = img.shape
        lbp_map = np.zeros_like(img, dtype=np.uint8)
        
        # Coordinates of the 8 neighbors (relative to the center pixel)
        
        
        neighbors_coords = [
            (0, 1),    
            (-1, 1),   
            (-1, 0),   
            (-1, -1),  
            (0, -1),   
            (1, -1),  
            (1, 0),    
            (1, 1)     
        ]

        # Iterate over the greyscale images, excluding the 1-pixel border 

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center_value = img[i, j]
                lbp_code = 0
                
                # Compare center pixel with its 8 neighbors
                for k, (dr, dc) in enumerate(neighbors_coords):
                    neighbor_value = img[i + dr, j + dc]
                    
                    # Comparison: If center <= neighbor, set bit to 1 (as per slide: "otherwise, write 1")
                    if center_value <= neighbor_value:
                        # Set the k-th bit (2**k) in the LBP code
                        lbp_code |= (1 << k)

                lbp_map[i, j] = lbp_code
        
        # Normalize lbp to [0,1]
        lbp_map = lbp_map.astype(np.float32) / 255.0

        # Use pdfs and normalize to have a probability vector
        pLBP = histogram(lbp_map, n_bins=n_bins) 

        return pLBP.astype(np.float32)

def _multiscale_lbp(img: np.ndarray,  P: int = 8, R: int = 1) -> np.ndarray:
    """
    Compute the multiscale Local Binary Pattern (LBP) representation of a greyscale image. We only used skimage's implementation,
    as we prefer to test the different metrics and parameters rather than optimizing our own LBP code.

    """

    n_bins = 2**P-1
         

    lbp_map = feature.local_binary_pattern(img, P=P, R=R, method="default")
    lbp_map = lbp_map.astype(np.float32) / (2**P - 1)
    pLBP = histogram(lbp_map, n_bins=n_bins) 
    return pLBP.astype(np.float32)


def compute_LBP_descriptors(
    imgs: List[np.ndarray], 
    method: str = "simple-lbp",
    greyscale: bool = True,
    P: int = 8,
    R: int = 1,
    n_crops: int = 2,
    save_pkl: bool = False
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



    if greyscale:

        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    cropped_imgs = [_spatial_crop(im, n_crops) for im in imgs]
    descs = []

    if method == "simple-lbp":
        initial_descs = [[_simple_lbp(img= crop, cv2_implementation=True) for crop in cropped_img] for cropped_img in cropped_imgs]
    elif method == "multiscale-lbp":
        initial_descs = [[_multiscale_lbp(img= crop, P=P, R=R) for crop in cropped_img] for cropped_img in cropped_imgs]
    else:
        raise ValueError(f"Invalid method ({method}) for computing image descriptors!")

    for cropped_img in initial_descs:

            # Concatenate and normalize
            final_histogram = np.concatenate(cropped_img, axis=0) / (n_crops*n_crops)
            descs.append(final_histogram)

# Save descriptors to a pickle file
    if save_pkl:
        # Make directory if not setted up
        print("Saving descriptors...")
        os.makedirs(SCRIPT_DIR / "descriptors", exist_ok=True)

        write_pickle(descs, SCRIPT_DIR / "descriptors" / f"{method}_{greyscale}_greyscale_P:{P}_R:{R}.pkl")

    return descs
        
    
# Compute descriptors for benchmark
if __name__=="__main__":
    print("Reading BBDD images...")
    bbdd_imgs = read_images(SCRIPT_DIR.parent.parent / "BBDD")
    query_imgs = read_images(SCRIPT_DIR.parent / "qsd1_w3")
    gt = read_pickle(SCRIPT_DIR.parent / "qsd1_w3" / "gt_corresps.pkl")


    for n_crop in range(2,10):
        for method in lbp_testing["methods"]:
                for greyscale in lbp_testing["greyscale"]:
                    for P in lbp_testing["P"]:
                        for R in lbp_testing["R"]:

                            data_descriptor = compute_LBP_descriptors(bbdd_imgs, method=method, greyscale=greyscale, P=P, R=R, save_pkl=False)
                            query_descriptor = compute_LBP_descriptors(query_imgs, method=method, greyscale=greyscale, P=P, R=R, save_pkl=False)
                            similarities = compute_similarities(query_descriptor, data_descriptor, metric="euclidean")
                                
                            # Sort the similarities and obtain their indices
                            indices = np.argsort(similarities, axis=1)
                            sorted_sims = np.take_along_axis(similarities, indices, axis=1)

                            # Extract the best k results
                            results_indices = indices[:, :5]
                            results_similarities = sorted_sims[:, :5]
                            map_score = mean_average_precision(indices, gt, k=5)

                            print(f"MAP@{5} score: {map_score:.4f}, n_crop: {n_crop}, using {P} P, {R} R.")

                            #Saving in txt file
                            with open(SCRIPT_DIR / "outputs" / f"lbp_cropped_map_scores.txt", "a") as f:
                                f.write(f"MAP@{5} score: {map_score:.4f}, n_crop: {n_crop}, using {P} P, {R} R.\n")

