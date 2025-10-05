"Computes RGB/HS/HSV (and RGB+HS) 1D color-histogram descriptors for images read from BBDD."
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


def _desc_rgb_hsv(rgb: np.ndarray, n_bins: int = 32, use_value: bool = False) -> np.ndarray:
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


def compute_descriptors(imgs: List[np.ndarray], 
                        method: str = "rgb",
                        n_bins: int = 32,
                        save_pkl: bool = False) -> List[np.ndarray]:

    # Baseline
    if method == "rgb":
        print("Computing RGB descriptors.")
        descs = [_desc_rgb(im, n_bins) for im in imgs]
    
    # Much stronger, H and S capture color independent of brightness
    elif method == "hs":
        print("Computing HS descriptors.")
        descs = [_desc_hsv(im, n_bins) for im in imgs]
        
    # Try to add the brightness
    elif method == "hsv":
        print("Computing HSV descriptors.")
        descs = [_desc_hsv(im, n_bins, use_value=True) for im in imgs]

    # Combine RGB and HS descriptors
    elif method == "rgb-hs":
        print("Computing RGB+HS descriptors.")
        descs = [_desc_rgb_hsv(im, n_bins) for im in imgs]
    
    else:
        raise ValueError(f"Invalid method ({method}) for computing image descriptors!")
    
    # Save descriptors to a pickle file
    if save_pkl:
        # Make directory if not setted up
        os.makedirs(SCRIPT_DIR / "descriptors", exist_ok=True)
        
        # write_pickle(descs, SCRIPT_DIR / f"{method}_{n_bins}bins_descriptors.pkl")
        write_pickle(descs, SCRIPT_DIR / "descriptors" / f"{method}_{n_bins}bins_descriptors.pkl")

    return descs
        
    
# Compute descriptors for benchmark
if __name__=="__main__":
    bbdd_imgs = read_images(SCRIPT_DIR.parent / "BBDD")

    for method in experiments["methods"]:
        for n_bins in experiments["n_bins"]:
            compute_descriptors(bbdd_imgs, method=method, n_bins=n_bins, save_pkl=True)
