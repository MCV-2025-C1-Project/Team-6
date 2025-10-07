import os
from typing import List

import numpy as np
from pathlib import Path

from params import experiments
from color_spaces import rgb_to_hsv
from histogram import histogram
from io_utils import write_pickle, read_images


SCRIPT_DIR = Path(__file__).resolve().parent

# Helpers
def _desc_rgb(rgb: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """
    RGB 1D histogram per channel.
    """
    rgb = rgb.astype(np.float32)

    # Normalize the channels
    R = rgb[..., 0] / 255.0
    G = rgb[..., 1] / 255.0
    B = rgb[..., 2] / 255.0

    # Use pdfs and normalize to have a probability vector
    pR = histogram(R, n_bins=n_bins) / 3
    pG = histogram(G, n_bins=n_bins) / 3
    pB = histogram(B, n_bins=n_bins) / 3

    return np.concatenate([pR, pG, pB], axis=0).astype(np.float32)


def _desc_hsv(rgb: np.ndarray, n_bins: int = 32, use_value: bool = False) -> np.ndarray:
    """
    HSV 1D histogram descriptor.
    By default use H and S. If use_value=True, adds V channel as well.
    """
    # From RGB to HSV
    hsv = rgb_to_hsv(rgb).astype(np.float32)

    # Normalize the channels (Hue from [0,360) to [0,1) range)
    H = hsv[..., 0] / 360.0
    S = hsv[..., 1]
    if use_value:
        V = hsv[..., 2]

    # Use channels pdfs
    pH = histogram(H, n_bins=n_bins)
    pS = histogram(S, n_bins=n_bins)
    if use_value:
        pV = histogram(V, n_bins=n_bins)
    concat_hists = [pH, pS]
    if use_value:
        concat_hists.append(pV)

    # Normalize to have a probability vector
    return np.concatenate(concat_hists, axis=0).astype(np.float32) / len(concat_hists)



def _desc_rgb_hsv(rgb: np.ndarray, n_bins: int = 32,  use_value: bool = False) -> np.ndarray:
    """
    RGB+HSV 1D histogram descriptor.
    By default use H and S. If use_value=True, adds V channel as well.
    """
    hsv_desc_len = 3 if use_value else 2

    # Obtain unnormalized RGB and HSV descriptors
    rgb_desc = _desc_rgb(rgb, n_bins) * 3
    hsv_desc = _desc_hsv(rgb, n_bins, use_value=use_value) * hsv_desc_len

    # Concatenate and normalize them jointly
    return np.concatenate([rgb_desc, hsv_desc]) / (3 + hsv_desc_len)

def _spatial_crop(img: np.ndarray, n_crops: int = 3, ) -> np.ndarray:
    crops = []
    h, w = img.shape[:2]

    crop_h, crop_w = h // n_crops, w // n_crops
    for i in range(n_crops):
        for j in range(n_crops):
            crops.append(img[i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w, :])
    return crops
        

def compute_spatial_descriptors(imgs: List[np.ndarray], 
                                n_crops: int = 3,
                                
                                pyramid: bool = False,
                                pyramid_levels: list = [1, 3,5],
                        method: str = "hsv",
                        n_bins: int = 16,
                        save_pkl: bool = False) -> List[np.ndarray]:
    if pyramid:
        descs = []
        img_count = len(imgs)
        level_descs = []
        for level in pyramid_levels:
            cropped_imgs = [_spatial_crop(im, level) for im in imgs]
            
            # Baseline
            if method == "rgb":
                initial_descs = [[_desc_rgb(crop, n_bins) for crop in cropped_img] for cropped_img in cropped_imgs]
                
            # Much stronger, H and S capture color independent of brightness
            elif method == "hs":
                initial_descs = [[_desc_hsv(crop, n_bins) for crop in cropped_img] for cropped_img in cropped_imgs]

                
            # Try to add the brightness
            elif method == "hsv":
                initial_descs = [[_desc_hsv(crop, n_bins,use_value=True) for crop in cropped_img] for cropped_img in cropped_imgs]


            # Combine RGB and HS descriptors
            elif method == "rgb-hs":
                initial_descs = [[_desc_rgb_hsv(crop, n_bins) for crop in cropped_img] for cropped_img in cropped_imgs]


            # Combine RGB and HSV descriptors
            elif method == "rgb-hsv":
                initial_descs = [[_desc_rgb_hsv(crop, n_bins,use_value=True) for crop in cropped_img] for cropped_img in cropped_imgs]

            
            else:
                raise ValueError(f"Invalid method ({method}) for computing image descriptors!")
            
            for cropped_img in initial_descs:
                    final_histogram = np.concatenate(cropped_img, axis=0)
                    level_descs.append(final_histogram)

        for img in range(img_count):
            level_descs_img = []
            for level in range( len(pyramid_levels)):
                level_descs_img.append(level_descs[img + img_count * level])
            descs.append(np.concatenate(level_descs_img, axis=0))
        
        if save_pkl:
            # Make directory if not setted up
            os.makedirs(SCRIPT_DIR / "descriptors", exist_ok=True)
            
            write_pickle(descs, SCRIPT_DIR / "descriptors" / f"{method}_{n_bins}_pyramid_descriptors.pkl")
    else: 
        cropped_imgs = [_spatial_crop(im, n_crops) for im in imgs]
        descs = []
        # Baseline
        if method == "rgb":
            initial_descs = [[_desc_rgb(crop, n_bins) for crop in cropped_img] for cropped_img in cropped_imgs]
            
        # Much stronger, H and S capture color independent of brightness
        elif method == "hs":
            initial_descs = [[_desc_hsv(crop, n_bins) for crop in cropped_img] for cropped_img in cropped_imgs]

            
        # Try to add the brightness
        elif method == "hsv":
            initial_descs = [[_desc_hsv(crop, n_bins,use_value=True) for crop in cropped_img] for cropped_img in cropped_imgs]


        # Combine RGB and HS descriptors
        elif method == "rgb-hs":
            initial_descs = [[_desc_rgb_hsv(crop, n_bins) for crop in cropped_img] for cropped_img in cropped_imgs]


        # Combine RGB and HSV descriptors
        elif method == "rgb-hsv":
            initial_descs = [[_desc_rgb_hsv(crop, n_bins,use_value=True) for crop in cropped_img] for cropped_img in cropped_imgs]

        
        else:
            raise ValueError(f"Invalid method ({method}) for computing image descriptors!")
        
        for cropped_img in initial_descs:
                # Concatenate and normalize
                final_histogram = np.concatenate(cropped_img, axis=0) / (n_crops*n_crops)
                descs.append(final_histogram)
    # Save descriptors to a pickle file
        if save_pkl:
            # Make directory if not setted up
            os.makedirs(SCRIPT_DIR / "descriptors", exist_ok=True)
            
            write_pickle(descs, SCRIPT_DIR / "descriptors" / f"{method}_{n_bins}bins_{n_crops}crops_noWeights_descriptors.pkl")

    return descs
        
    
# Compute descriptors for benchmark
if __name__=="__main__":
    bbdd_imgs = read_images(SCRIPT_DIR.parent / "BBDD")
    for n_crop in experiments["n_crops"]:
        print(f"Computing {n_crop} crops descriptors...")
        compute_spatial_descriptors(bbdd_imgs,pyramid=False, method="hsv", n_bins=16, save_pkl=True)

    
                
