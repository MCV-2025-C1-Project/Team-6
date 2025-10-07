import numpy as np
import matplotlib.pyplot as plt

from sklearn.covariance import MinCovDet
from scipy.stats import chi2
from color_spaces import rgb_to_hsv
from typing import List
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure

def extract_border_samples(img: np.ndarray, border_width: int = 10) -> np.ndarray:
    """
    Extracts the border pixels of the image as color samples.
    Args:
        img: Input image from which to extract border samples.
        border_width: Width of the border to extract (default is 10 pixels).

    Returns:
        A 2D array of shape (N, C) containing the color samples from the border pixels.
    """

    top = img[:border_width, :]
    bottom = img[-border_width:, :]
    left = img[:, :border_width]
    right = img[:, -border_width:]
    samples = np.vstack([
        top.reshape(-1, img.shape[2]),
        bottom.reshape(-1, img.shape[2]),
        left.reshape(-1, img.shape[2]),
        right.reshape(-1, img.shape[2]),
    ])
    return samples  # shape (N, C)


def remove_background(images: List[np.ndarray], border_width: int = 10):

    masks = []

    for i, img in enumerate(images):

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original")

        # First simple option: use color samples from the corners of the images
        # with HS color space (without the value V)
        hsv = rgb_to_hsv(img)
        hs = hsv[..., :2] 

        # Normalization (Hue from [0, 360])
        hs[..., 0] /= 360

        # Border with of the samples 10 pix by default
        samples = extract_border_samples(hs, border_width=border_width)

        # Minimum Covariance Determinant for robust estimation of background color distr
        cov_fraction = 0.75 # Recommended 
        mcd = MinCovDet(support_fraction=cov_fraction, random_state=0).fit(samples)
        mean = mcd.location_
        cov = mcd.covariance_

        # Moore-Penrose pseudo-inverse to avoid problems such as covariance matrix being singular (non-invertible)
        cov_inv = np.linalg.pinv(cov)

        # Computation of Mahalanobis distance for every pixel
        # Distance between a point P and a probability distr (inv-cov)
        H, W = hs.shape[:2]
        flat = hs.reshape(-1, 2)

        # Substraction of the background mean from every pixel color (point-i - mean)
        delta = flat - mean
        d2 = np.einsum('ij,ij->i', delta @ cov_inv, delta)
        d2 = d2.reshape(H, W)

        # Threshold using chi-square
        # 2 degrees of freedom and critical value at 0.01 significance level
        threshold = chi2.ppf(1-0.01, 2)
        mask_bg = d2 <= threshold 

        # p_values = chi2.sf(d2, df=2)  # survival function P(X > D^2)
        # alpha = 0.01                  # significance level

        # mask_bg = p_values > alpha 

        struct = generate_binary_structure(2, 2)
        mask_bg = binary_opening(mask_bg, structure=struct)
        mask_bg = binary_closing(mask_bg, structure=struct)

        plt.subplot(1, 2, 2)
        plt.imshow(~mask_bg, cmap="gray")
        plt.title("Foreground (painting)")
        plt.show()
        plt.close()

        masks.append(mask_bg.astype(bool))
    
    return masks