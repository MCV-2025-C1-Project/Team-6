from typing import List

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.ndimage import (binary_opening, 
                           binary_closing, 
                           binary_propagation, 
                           binary_fill_holes,
                           median_filter,
                           label)
from skimage.morphology import convex_hull_image
from color_spaces import rgb_to_hsv, rgb_to_lab


def extract_border_samples(img: np.ndarray, border_width: int = 20) -> np.ndarray:
    """
    Extracts border pixels using a boolean mask to avoid double-counting corners.
    Args:
        img: Input image from which to extract border samples.
        border_width: Width of the border to extract (default is 10 pixels).

    Returns:
        A 2D array of shape (N, C) containing the color samples from the border pixels.
    """

    h, w, _ = img.shape

    # Mask of booleans
    mask = np.zeros((h, w), dtype=bool)
    
    # Set the border regions to True
    mask[:border_width, :] = True
    mask[-border_width:, :] = True
    mask[:, :border_width] = True
    mask[:, -border_width:] = True
    
    # Mask is gonna be used to grab all border pixels at once
    samples = img[mask]
    
    # Limit the number of samples to a maximum (to avoid too much computation)
    max_samples = 5000
    if len(samples) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = samples[idx]
        
    return samples  # shape (N, C)



def remove_background(images: List[np.ndarray], border_width: int = 10, color_space = "lab",
                      use_percentile_thresh=True, percentile=99):

    # List of masks of the images
    masks = []

    for _ , img in enumerate(images):
        
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Original")

        # Locally smooth the color noise (so we get better results with the masks near the borders!)
        img_smooth = median_filter(img, size=(3, 3, 1))
        
        # Try a*, b* (LAB) or HS (HSV)
        if color_space == "lab":
            lab = rgb_to_lab(img_smooth)
            channels = lab[..., 1:3]
        else:
            hsv = rgb_to_hsv(img_smooth)
            channels = hsv[..., :2] 
            # Normalization (Hue from [0, 360])
            channels[..., 0] /= 360

        # Border taken from the samples, 20 pix by default
        samples = extract_border_samples(channels, border_width=border_width)

        # Minimum Covariance Determinant -> remove a fraction of outliers (more robust than simple covariance)
        cov_fraction = 0.75
        mcd = MinCovDet(support_fraction=cov_fraction, random_state=0).fit(samples)
        mean = mcd.location_
        cov = mcd.covariance_

        # Moore-Penrose pseudo-inverse to avoid problems such as covariance matrix being singular (non-invertible)
        cov_inv = np.linalg.pinv(cov)

        # Computation of Mahalanobis distance for every pixel
        # Distance between a point P and a probability distr (inv-cov)
        H, W = channels.shape[:2]
        flat = channels.reshape(-1, 2)

        # Substraction of the background mean from every pixel color (point-i - mean)
        delta = flat - mean
        d2 = np.einsum('ij,ij->i', delta @ cov_inv, delta).reshape(H, W)

        if use_percentile_thresh:
            # Threshold based on the samples of the border
            # We compute the Mahalanobis distance of the border samples to get a threshold
            delta_s = samples - mean
            d2_s = np.einsum('ij,ij->i', delta_s @ cov_inv, delta_s)
            threshold = np.percentile(d2_s, percentile)
        else:
            # More robust...we do not have a perfect Gaussian distribution!
            threshold = chi2.ppf(1 - 0.01, df=2)

        mask_bg = (d2 <= threshold)

        foreground = ~mask_bg

        struct = np.ones((5, 5), dtype=bool)   # 5x5 (cuadrado)
        foreground = binary_opening(foreground, structure=struct, iterations=1)
        foreground = binary_closing(foreground, structure=struct, iterations=1)

        plt.subplot(1, 3, 2)
        plt.imshow(foreground, cmap="gray")
        plt.title("Candidate background")


        # Label all disconnected regions in the foreground
        struct = np.ones((3, 3), dtype=bool)   
        labeled_mask, num_labels = label(foreground, structure=struct)

        final_mask = np.zeros_like(foreground, dtype=bool)
        if num_labels > 0:
            counts = np.bincount(labeled_mask.ravel())
            largest_component_label = counts[1:].argmax() + 1
            final_mask = (labeled_mask == largest_component_label)
        else:
            final_mask = foreground
    
        foreground = convex_hull_image(final_mask)

        # Propagate background only from border
        # TODO: should we code this from scratch?
        # seed = np.zeros_like(candidate_bg, dtype=bool)
        # bw = border_width
        # seed[:bw, :] = True; seed[-bw:, :] = True
        # seed[:, :bw] = True; seed[:, -bw:] = True

        # background = binary_propagation(seed, mask=candidate_bg)

        # foreground = ~background
        # org_mask = binary_fill_holes(foreground)                 # light cleanup
        # Optional light smoothing:
        # foreground = binary_closing(fg, structure=struct, iterations=1)


        plt.subplot(1, 3, 3)
        plt.imshow(foreground, cmap="gray")
        plt.title("Foreground (painting)")
        plt.show(); plt.close()

        masks.append(foreground.astype(bool))
    
    return masks

# TODO: the algorithm seems quite good, now we need just small ajustments.
# Also I think it it time to code the pipeline of creating the masks and testing to compute
# f1, recall, precision...

# TODO:  Do we need the median filter at the beggining? how many iterations for opening and closing?
# what percentile for thresold or chi2? what cov fraction? test them so we get the best params