"""This script removes noise from pictures by using filters """
import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import uniform_filter, median_filter

from utils.io_utils import read_images

SCRIPT_DIR = Path(__file__).resolve().parent

# here we'll use YCrCb, much better for noise removal

# Helpers
# Filters
def mean_filter(image: np.ndarray, kernel_size: tuple = (3, 3, 1)) -> np.ndarray:
    """Replaces each pixel with the average of its neighbors.
    Reduces uncorrelated Gaussian noise. Blurs edges and small details and
    is not robust to outliers.
    """
    return uniform_filter(image, size=kernel_size)

def median_filter(image: np.ndarray, kernel_size: int = 3) ->  np.ndarray:
    """Take the median of the piels udner the kernel.
    Non-linear: it replaces outliers but keeps step edges sharp.
    Ineffective for Gaussian or structured noise."""
    return cv2.medianBlur(image, kernel_size)

def gaussian_blur(image: np.ndarray, kernel_size: tuple = (15, 15)) ->  np.ndarray:
    """Smooths random Gaussian noise while preserving edges a bit better than uniform averaging.
    Still blurs edges and not robust to outliers.
    """
    return cv2.GaussianBlur(image, kernel_size , 0)

#Bilateral Median Filter
#Adaptive Median Filter


# Ratios to check noise in images
def salt_and_pepper_ratio(u8_gray_image: np.ndarray, t: int = 2) -> float:
    """Fraction of pixels that are very close to 0 or 255 (extreme intensity outliers).
    Used for removing black and white specks. It is non-Gaussian. 
    Low ratio (<0.5%) → image is probably clean.
    Moderate (1–3%) → sparse impulse noise.
    High (>5%) → heavy impulse corruption.
    Fix with median filter."""
    near_low  = (u8_gray_image <= t).mean()
    near_high = (u8_gray_image >= 255 - t).mean()
    return float(near_low + near_high) 

def _laplace_sigma(gray_image: np.ndarray) -> float:
    """Robust estimate of Gaussian-like noise variance.
    Low σ (<0.008–0.010): clean.
    Medium σ (≈0.015): mild Gaussian/Poisson.
    High σ (>0.02): very noisy.
    Fixed by Mean filter / Gaussian blur / NLM / BM3D.
    """
    lap = cv2.Laplacian(gray_image, cv2.CV_32F, ksize=3)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    return float(mad / 0.6745)

def _jpeg_blockiness(u8_gray_image: np.ndarray) -> float:
    """Compares mean absolute differences across 8×8 JPEG block boundaries versus within blocks.
    If boundaries have significantly higher jumps, the image has compression artifacts.
    Ratio ≈ 1 → no block artifacts.
    1.1–1.2 → visible blocking.
    Fixed by Bilateral or Non-Local Means.
    """
    h, w = u8_gray_image.shape
    # diffs along 8x8 boundaries vs. interior
    v_edges = np.mean(np.abs(np.diff(u8_gray_image[:, 7::8], axis=1))) if w >= 16 else 0.0
    h_edges = np.mean(np.abs(np.diff(u8_gray_image[7::8, :], axis=0))) if h >= 16 else 0.0
    interior = (
        np.mean(np.abs(np.diff(u8_gray_image, axis=1))) +
        np.mean(np.abs(np.diff(u8_gray_image, axis=0)))
    ) + 1e-6
    return float((v_edges + h_edges) / interior)




# NOT DONE YET!
def denoise_always(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    Y_med = cv2.medianBlur(Y, 3)                                # 3x3 median
    Y_bil = cv2.bilateralFilter(Y_med, d=5, sigmaColor=10, sigmaSpace=3)

    # Fast NLM on chroma (OpenCV expects single-channel 8-bit)
    Cr_nlm = cv2.fastNlMeansDenoising(Cr, h=5, templateWindowSize=7, searchWindowSize=21)
    Cb_nlm = cv2.fastNlMeansDenoising(Cb, h=5, templateWindowSize=7, searchWindowSize=21)

    out = cv2.merge([Y_bil, Cr_nlm, Cb_nlm])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dir1', '--data-dir1',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd1_w3",
        help='Path to a directory of images with noise.'
    )

    dir1 = parser.parse_args().data_dir1
    if not dir1.is_dir():
        raise ValueError(f"{dir1} is not a valid directory.")
    
    images = read_images(dir1)

    for image in images[0:2]:
        print(image)
        print(image.shape)
        print(type(image))
        #denoised_image = denoise_always(image)
        #final_image = np.concatenate((image, denoised_image), axis=1)
        #cv2.imshow('Noise removal', final_image); cv2.waitKey(0)
    cv2.destroyAllWindows()
