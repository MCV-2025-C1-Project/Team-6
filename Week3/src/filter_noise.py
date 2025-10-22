"""
This script removes noise from pictures by using filters. 
Main can execute the grid search or the evaluation of the development set.
"""

import argparse
from tqdm import tqdm
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from params import noise_search_space, base_thresholds, best_noise_params
from utils.io_utils import read_images
from utils.color_spaces import rgb_to_ycrcb
from utils.plots import plot_all_comparisons

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_THRESHOLDS = base_thresholds
BEST_THRESHOLDS = best_noise_params  
SEARCH_SPACE = noise_search_space 


#########################################
################ Helpers ################
#########################################   

def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Convert any image to uint8 grayscale for metrics that expect gray_u8."""
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    if gray.dtype != np.uint8:
        gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        gray_u8 = gray
    return gray_u8


#######################################################
#################### Filters ##########################
#######################################################

def mean_filter(image: np.ndarray, kernel_size: tuple = (3, 3, 1)) -> np.ndarray:
    """
    Replaces each pixel with the average of its neighbors.
    Reduces uncorrelated Gaussian noise. Blurs edges and small details and is not robust to outliers.
    Args:
        image: Input image to filter.
        kernel_size: Size of the averaging kernel.
    Returns:
        Filtered image.
    """
    return uniform_filter(image, size=kernel_size)


def median_filter(image: np.ndarray, kernel_size: int = 3) ->  np.ndarray:
    """
    Take the median of the pixels under the kernel.
    Non-linear -> it replaces outliers but keeps step edges sharp.
    Args:
        image: Input image to filter.
        kernel_size: Size of the median filter kernel (must be odd).
    Returns:
        Filtered image.
    """
    return cv2.medianBlur(image, kernel_size)


def gaussian_blur(image: np.ndarray, kernel_size: tuple = (15, 15)) ->  np.ndarray:
    """
    Smooths random Gaussian noise while preserving edges a bit better than uniform averaging.
    Still blurs edges and not robust to outliers.
    Args:
        image: Input image to filter.
        kernel_size: Size of the Gaussian kernel.
    Returns:
        Filtered image.
    """
    return cv2.GaussianBlur(image, kernel_size , 0)


def bilateral_filter(image_luminance, d=5, sigma_c=13, sigma_s=3):
    """
    Smooths within regions but doesn’t cross strong edges.
    Used for Gaussian-ish noise or JPEG artifacts
    Args:
        image_luminance: Input image to filter (grayscale or single channel).
        d: Diameter of each pixel neighborhood.
        sigma_c: Filter sigma in color space.
        sigma_s: Filter sigma in coordinate space.
    Returns:
        Filtered image.
    """
    # Ensure uint8 input
    if image_luminance.dtype != np.uint8:
        image_luminance = np.clip(np.rint(image_luminance), 0, 255).astype(np.uint8)
    
    out = cv2.bilateralFilter(image_luminance, d=d, sigmaColor=sigma_c, sigmaSpace=sigma_s)
    assert out.shape == image_luminance.shape
    
    return out
    

def adaptive_median_filter(image: np.ndarray, Smax: int = 7) -> np.ndarray:
    """
    Removes impulse noise while preserving edges.
    Args:
        image: Input image to filter (grayscale or color).
        Smax: Maximum window size (must be odd and >= 3).
    Returns:
        Filtered image.
    """
    # Check Smax
    if Smax % 2 == 0 or Smax < 3:
        raise ValueError("Smax must be odd and >= 3")

    def _amf_gray(gray_u8: np.ndarray) -> np.ndarray:
        Y = gray_u8.copy()
        out = Y.copy()
        undecided = np.ones(Y.shape, dtype=bool)
        last_Zmed = None

        for S in range(3, Smax + 1, 2):
            Zmed = cv2.medianBlur(Y, S) # local median per pixel.
            k = np.ones((S, S), np.uint8)
            Zmin = cv2.erode(Y, k) # local minimum per pixel.
            Zmax = cv2.dilate(Y, k) # local maximum per pixel.

            # Zmin < Zmed < Zmax
            A = (Zmin < Zmed) & (Zmed < Zmax) & undecided
            if A.any():
                # keep center if Zmin < Y < Zmax else replace with Zmed
                keep = (Zmin < Y) & (Y < Zmax)
                sel_keep = A & keep
                sel_rep  = A & (~keep)
                out[sel_keep] = Y[sel_keep]
                out[sel_rep]  = Zmed[sel_rep]
                undecided[A] = False

            last_Zmed = Zmed
            if not undecided.any():
                break

        # For pixels where Stage A never succeeded by Smax: use Zmed at largest S
        if undecided.any():
            out[undecided] = last_Zmed[undecided]
        return out

    # Apply to grayscale / color
    is_u8 = image.dtype == np.uint8
    if image.ndim == 2:
        gray_u8 = image if is_u8 else _to_gray_u8(image)
        out = _amf_gray(gray_u8)
        return out if is_u8 else out.astype(image.dtype)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        chans = []
        for c in range(image.shape[2]):
            ch = image[..., c]
            gray_u8 = ch if ch.dtype == np.uint8 else _to_gray_u8(ch)
            chans.append(_amf_gray(gray_u8))
        out = np.stack(chans, axis=2)
        return out if is_u8 else out.astype(image.dtype)
    else:
        raise ValueError("image must be (H,W) or (H,W,C)")


######################################################
############### Ratios to check noise ################
######################################################

def salt_and_pepper_ratio(
    u8_gray_image: np.ndarray,
    t_extreme: int = 12, 
    win: int = 3, 
    delta_abs: int = 30, 
    k_mad: float = 3.5,
    flat_pct: float = 60.0,
    border: int = 8
) -> float:
    """
    Fraction of pixels that behave like isolated impulses.
    Combines (A) near-extreme tails and (B) local outliers vs. local median,
    evaluated mostly on flat regions (to avoid mistaking texture for impulses).
    Args:
        u8_gray_image: Input grayscale image (8-bit unsigned).
        t_extreme: How far we count as 'extreme' in tails.
        win: Local window (3 or 5) for median.
        delta_abs: Absolute jump to call an impulse.
        k_mad: Robust (MAD-based) threshold.
        flat_pct: Percentile for flatness detection.
        border: Border size to crop for analysis.

    Returns:
        Fraction of pixels behaving like isolated impulses.
    """
    # odd window
    if win % 2 == 0 or win < 3:
        raise ValueError("win must be odd and >= 3")
    
    # border crop to avoid frames biasing the metric
    if border > 0 and min(u8_gray_image.shape[:2]) > 2*border:
        g = u8_gray_image[border:-border, border:-border]
    else:
        g = u8_gray_image

    # Find near-extreme tails
    med3 = cv2.medianBlur(g, 3)
    dev  = np.abs(g.astype(np.int16) - med3.astype(np.int16))
    near = (g <= t_extreme) | (g >= 255 - t_extreme)
    tails_mask = near & (dev >= delta_abs)

    # Keep only isolated speck (discard lines adn edges)
    nbr_t = cv2.boxFilter(tails_mask.astype(np.uint8), -1, (3, 3), normalize=False) # sum of neighbours
    tails_mask = tails_mask & (nbr_t <= 2)
    tails = tails_mask.mean()

    # Find local outliers on flat areas
    med  = cv2.medianBlur(g, win)
    diff = np.abs(g.astype(np.int16) - med.astype(np.int16))
    mad  = cv2.medianBlur(diff.astype(np.uint8), win).astype(np.float32)
    gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    flat = mag <= np.percentile(mag, flat_pct) # keep only flattest pixels

    # impulse if diff (deviation) >= max(absolute threshold, k * MAD)
    local_thr = np.maximum(delta_abs, k_mad * (mad + 1.0))
    impulses = (diff.astype(np.float32) >= local_thr) & flat

    # keep only specks
    nbr_i = cv2.boxFilter(impulses.astype(np.uint8), -1, (3, 3), normalize=False)
    impulses = impulses & (nbr_i <= 2)
    sp_local = impulses.mean()

    return float(max(tails, sp_local))

def salt_and_pepper_ratio_rgb(
    rgb_u8: np.ndarray,
    win: int = 3,
    delta_abs: int = 8,      
    k_mad: float = 2.0,      
    flat_pct: float = 80.0, 
    iso_max: int = 2,        
    border: int = 4
) -> float:
    """
    Fraction of pixels behaving like isolated local outliers in ANY color channel.
    Robust to colored S&P dots.
    """
    assert rgb_u8.ndim == 3 and rgb_u8.shape[2] >= 3, "expects RGB uint8"
    if rgb_u8.dtype != np.uint8:
        g = cv2.normalize(rgb_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        g = rgb_u8

    if border > 0 and min(g.shape[:2]) > 2*border:
        g = g[border:-border, border:-border, :]

    H, W, _ = g.shape
    impulses_any = np.zeros((H, W), dtype=bool)

    # Edge/flatness mask computed on luminance
    Y = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m_th = np.percentile(mag, flat_pct)
    flat = (mag <= m_th)

    # Process each channel separately
    for c in range(3):
        ch = g[..., c]
        med = cv2.medianBlur(ch, win)
        diff = (ch.astype(np.int16) - med.astype(np.int16))
        adiff = np.abs(diff)

        # local MAD of |ch - med|
        mad = cv2.medianBlur(adiff.astype(np.uint8), win).astype(np.float32)

        # threshold
        thr = np.maximum(delta_abs, k_mad * (mad + 1.0))
        cand = (adiff.astype(np.float32) >= thr) & flat

        # isolation: at most iso_max flagged in 3x3
        nbr = cv2.boxFilter(cand.astype(np.uint8), -1, (3, 3), normalize=False)
        imp = cand & (nbr <= iso_max)

        impulses_any |= imp

    return float(impulses_any.mean())


def jpeg_blockiness(u8_gray_image: np.ndarray) -> float:
    """
    Low-quality JPEG creates visible jumps exactly at 8×8 block borders.
    If the borders have much larger jumps than the interior, the ratio > 1
    Args:
        u8_gray_image: Input grayscale image (8-bit unsigned).
    Returns:
        Blockiness ratio: ~1.0 for clean, >1.1 for blocky
    """
    # Ensure 2D grayscale
    if u8_gray_image.ndim != 2:
        raise ValueError("jpeg_blockiness expects a 2D grayscale image.")
    h, w = u8_gray_image.shape

    g = u8_gray_image.astype(np.float32)

    # pair 7|8, 15|16, ... so shapes match
    c7 = g[:, 7:w-1:8]
    c8 = g[:, 8:w:8]
    #mean absolute vertical jump across boundaries c7 and c8
    v_bound = np.median(np.abs(c8 - c7)) if c7.size and c8.size else 0.0

    r7 = g[7:h-1:8, :]
    r8 = g[8:h:8, :]
    #mean absolute horizontal jump across boundaries r7 and r8
    h_bound = np.median(np.abs(r8 - r7)) if r7.size and r8.size else 0.0

    # interior adjacent diffs, excluding boundary pairs
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    
    if w > 1:
        mask_h = np.ones_like(dh, dtype=bool)
        mask_h[:, 7::8] = False
        dh_int = dh[mask_h].mean()
    else:
        dh_int = 0.0
    
    if h > 1:
        mask_v = np.ones_like(dv, dtype=bool)
        mask_v[7::8, :] = False
        dv_int = dv[mask_v].mean()
    else:
        dv_int = 0.0

    interior = (dh_int + dv_int)
    
    # scene too flat to judge
    if interior < 1e-2:
        return 1.0
    
    return float((v_bound + h_bound) / (interior + 1e-6))


def flat_region_sigma(gray_u8: np.ndarray, flat_pct: float = 15.0, var_win: int = 7) -> float:
    """
    Robust estimate of Gaussian-like noise, measured on truly flat pixels.
    Args:
        gray_u8: Input grayscale image (8-bit unsigned).
        flat_pct: Percentile of flattest pixels to keep (stricter than 30%).
        var_win: Window size for local variance (should be odd).
    Returns:
        Estimated sigma of Gaussian-like noise in [0,1] intensity units.
    """
    g0 = cv2.medianBlur(gray_u8, 3)
    g = g0.astype(np.float32)

    # Local variance (texture detector) via box filters
    mu  = cv2.boxFilter(g, -1, (var_win, var_win))
    mu2 = cv2.boxFilter(g * g, -1, (var_win, var_win))
    var = np.maximum(0.0, mu2 - mu * mu)

    # Gradient magnitude (edge detector)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Keep only pixels that are both low-variance AND low-gradient
    v_th = np.percentile(var, flat_pct)
    m_th = np.percentile(mag, flat_pct)
    flat = (var <= v_th) & (mag <= m_th)
    
    if not np.any(flat):
        return 0.0

    # MAD of Laplacian on flat pixels
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    vals = lap[flat]
    med  = np.median(vals)
    mad  = np.median(np.abs(vals - med)) + 1e-6

    return float((mad / 0.6745) / 255.0)


def variance_of_laplacian(img: np.ndarray, ksize: int = 3) -> float:
    """
    Variance of Laplacian.
    Higher mean sharper (Laplacian has big swings because of lots of high-frequency energies)
    Very low mean likely out-of-focus blur (low variance of Laplacian)
    Args:
        img: Input image (grayscale or color).
        ksize: Kernel size for Laplacian.
    Returns:
        Variance of the Laplacian.
    """
    gray_u8 = _to_gray_u8(img)
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F, ksize=ksize)
    return float(lap.var())


def chroma_noise_ratio(rgb: np.ndarray) -> float:
    """
    Simple high-pass residual for each chroma channels.
    Many cameras produce color speckle: fine, grainy noise mostly in chroma
    Args:
        rgb: Input RGB image.
    Returns:
        Fraction of chroma energy that is high-frequency.
    """

    ycrcb = rgb_to_ycrcb(rgb)
    _, Cr, Cb = cv2.split(ycrcb)
    
    # remove low-freq (3x3 mean), measure residual energy
    Cr_h = Cr.astype(np.float32) - cv2.blur(Cr, (3,3)).astype(np.float32) # blur is a low-pass version of Cr
    Cb_h = Cb.astype(np.float32) - cv2.blur(Cb, (3,3)).astype(np.float32)
    num = Cr_h.var() + Cb_h.var() #variance of that high-pass residual
    den = (Cr.astype(np.float32).var() + Cb.astype(np.float32).var() + 1e-6)
    
    # return fraction of chroma energy that is high-frequency.
    return float(num / den)  # higher → more high-freq chroma noise


######################################################
#################### Main methods ####################
######################################################

def measure_noise_metrics(img: np.ndarray) -> dict:
    """
    Compute core metrics used for blind noise detection.
    Args:
        img: Input image to analyze.
    Returns:
        A dict with the computed metrics.
         - sp: salt-and-pepper ratio (0..1)
         - sig: flat-region sigma (in [0,1] intensity units)
         - blk: JPEG blockiness ratio (~1 clean, >1.1 blocky)
         - vol: variance of Laplacian (sharpness proxy)
         - chr: chroma noise ratio (only for color images; None otherwise)
    """
    # Convert to grayscale u8 for metrics that need it
    gray_u8 = _to_gray_u8(img)

    if img.ndim == 3 and img.shape[2] >= 3:
        sp = salt_and_pepper_ratio_rgb(img)  
        chrn = chroma_noise_ratio(img)

    else:
        sp = salt_and_pepper_ratio(gray_u8) 
        chrn = None

    # Compute metrics
    sig = flat_region_sigma(gray_u8)         
    blk = jpeg_blockiness(gray_u8)
    vol = variance_of_laplacian(img)
    return {"sp": sp, "sig": sig, "blk": blk, "vol": vol, "chr": chrn}


def check_noise(img: np.ndarray, thresholds: dict | None = None, margin = 0.002) -> list[str]:
    """
    Classify noise types present in the image.
    Args:
        img: Input image to analyze.
        thresholds: Optional dict of thresholds for each noise type.
        margin: Small margin to avoid borderline cases.
    Returns:
        A list like: ["impulse", "gaussian_like", "jpeg_blockiness", "chroma_noise"]
        If nothing triggers, returns ["clean"].
    """
    
    # Default thresholds
    # Normally from params.py, but this allows easy overrides
    thr = {
        "sp_impulse": 0.08,   
        "sig_gauss":  0.05,  
        "blk_jpeg":   1.15,    
        "chr_chroma": 0.25,    
        "vol_blur":   80.0   
    }  

    if thresholds:
        thr.update(thresholds)

    m = measure_noise_metrics(img)
    labels: list[str] = []

    sp_hit  = (m["sp"]  >= thr["sp_impulse"] + margin)
    sig_hit = (m["sig"] >= thr["sig_gauss"]  + margin)

    # Impulse noise (salt & pepper)
    if sp_hit:
        labels.append("impulse")

    # Gaussian-like / random noise
    if sig_hit and not sp_hit:
        labels.append("gaussian_like")
    
    elif sig_hit and sp_hit and m["sig"] >= 2.5 * thr["sig_gauss"]:
        labels.append("gaussian_like")

    # JPEG blocking artifacts
    if m["blk"] >= thr["blk_jpeg"]:
        labels.append("jpeg_blockiness")

    # Chroma speckle (only if color metric available)
    if (m["chr"] is not None) and (m["chr"] >= thr["chr_chroma"]):
        labels.append("chroma_noise")

    if not labels:
        labels = ["clean"]

    return labels


def denoise_images(image: np.ndarray, thresholds: dict | None = None, return_labels: bool = False) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """
    Denoise a single image based on detected noise types.
    Args:
        image: Input image to denoise.
        thresholds: Optional dict of thresholds for noise detection.
        return_labels: If True, returns a tuple (denoised_image, labels).
    Returns:
        Denoised image (or tuple if return_labels is True).
    """

    # Detect noise types
    labels = check_noise(image, thresholds)

    # clean-case
    if labels == ["clean"]:
        return (image, labels) if return_labels else image
    
    # gray-case
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        out = image.copy()
        # Apply filters based on detected noise types
        if "impulse" in labels:
            out = adaptive_median_filter(out, Smax=7)
            out = median_filter(out, kernel_size=3)
        elif ("gaussian_like" in labels) or ("jpeg_blockiness" in labels):
            out = bilateral_filter(out, d=5, sigma_c=12, sigma_s=3)
        else:
            out = image  # clean
        return (out, labels) if return_labels else out
    
    # 3 channel case
    ycrcb = rgb_to_ycrcb(image)
    Y, Cr, Cb = cv2.split(ycrcb)
    outY, outCr, outCb = Y.copy(), Cr.copy(), Cb.copy()
    
    # Apply filters based on detected noise types
    if "impulse" in labels:
        outY = adaptive_median_filter(outY, Smax=7)
        outY = median_filter(outY, kernel_size=3)

    elif ("gaussian_like" in labels) or ("jpeg_blockiness" in labels):
        outY = bilateral_filter(outY, d=5, sigma_c=12, sigma_s=3)

    if "chroma_noise" in labels:
        outCr = bilateral_filter(outCr, d=7, sigma_c=15, sigma_s=4)
        outCb = bilateral_filter(outCb, d=7, sigma_c=15, sigma_s=4)

    # Merge channels back
    out = cv2.merge([outY, outCr, outCb])
    out = cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)

    return (out, labels) if return_labels else out


def denoise_batch(images: list[np.ndarray], thresholds: dict = None, return_labels: bool = False) -> list[np.ndarray]:
    """
    Denoise a batch of images.
    Args:
        images: List of images to denoise.
        thresholds: Optional dict of thresholds for noise detection.
        return_labels: If True, returns a list of tuples (denoised_image, labels).
    Returns:
        List of denoised images (or tuples if return_labels is True).
    """
    outs = []

    for img in tqdm(images, desc="Denoising images..."):
        den = denoise_images(img, thresholds=thresholds, return_labels=return_labels)
        outs.append(den)
    return outs

##############################################
############# Evaluation Methods #############
##############################################

def eval_psnr_ssim(orig: np.ndarray, den: np.ndarray, gt: np.ndarray) -> float:
    """
    Evaluate denoising quality using PSNR and SSIM improvements.
    Args:
        orig: Original noisy image.
        den: Denoised image.
        gt: Ground truth image.
    Returns:
        Combined score based on PSNR and SSIM improvements.
    """
    # Convert to grayscale u8
    orig, den, gt = _to_gray_u8(orig), _to_gray_u8(den), _to_gray_u8(gt)

    # Resize images to match ground truth
    H,W = gt.shape[:2]
    if orig.shape[:2] != (H,W): 
        orig = cv2.resize(orig, (W,H), cv2.INTER_AREA)
    if den.shape [:2] != (H,W): 
        den  = cv2.resize(den , (W,H), cv2.INTER_AREA)

    # Compute PSNR and SSIM
    if np.array_equal(orig, gt):
        return 0.0

    psnr_og = psnr(gt, orig, data_range=255)
    ssim_og = ssim(gt, orig, data_range=255)
    psnr_den = psnr(gt, den, data_range=255)
    ssim_den = ssim(gt, den, data_range=255)

    d_psnr = psnr_den - psnr_og
    d_ssim = ssim_den - ssim_og
    eps_psnr, eps_ssim = 0.05, 0.001

    # Combined score
    score = max(0.0, d_ssim - eps_ssim) - max(0.0, -(d_ssim) - eps_ssim)
    score += 0.05 * (max(0.0, d_psnr - eps_psnr) - max(0.0, -(d_psnr) - eps_psnr))

    return score

def eval_batch(og_images: list[np.ndarray], den_images: list[np.ndarray], gts: list[np.ndarray]) -> tuple[list[float], float]:
    """
    Evaluate a batch of denoised images.
    Args:
        og_images: List of original noisy images.
        den_images: List of denoised images.
        gts: List of ground truth images.
    Returns:
        A tuple (scores, mean_score) where scores is a list of individual scores and mean_score is the average score.
    """ 
    scores = []

    for o, d, g in zip(og_images, den_images, gts):
        s = eval_psnr_ssim(o, d, g)
        scores.append(float(s))
    mean_score = float(np.mean(scores)) if scores else 0.0
    return scores, mean_score


def grid_search_thresholds(og: list[np.ndarray], gts: list[np.ndarray], base_thr: dict, search_space: dict) -> tuple[dict, float]:
    """
    Grid search over noise detection thresholds to maximize denoising score.
    
    Args:
        og: List of original noisy images.
        gts: List of ground truth images.
        base_thr: Base thresholds to start from.
        search_space: Dictionary defining the search space for each threshold.
    Returns:
        best_thr: Dictionary of best thresholds found.
        best_score: Best mean score achieved.
    """
    # Initialize bests
    best_thr = base_thr
    best_score = -1e9

    # Get keys and values
    keys = list(search_space.keys())
    value_list = [search_space[k] for k in keys]

    for vals in product(*value_list):
        # Create new threshold dict
        thr = {**base_thr, **dict(zip(keys, vals))}

        # Denoise and evaluate
        dens = denoise_batch(og, thresholds=thr, return_labels=False)
        _, mean_score = eval_batch(og, dens, gts)

        # Update bests if needed
        if mean_score > best_score:
            best_score = mean_score
            best_thr = thr

    return best_thr, best_score



if __name__=="__main__":

    # Define parsers for data paths
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dir1', '--data-dir1',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd1_w3",
        help='Path to a directory of images with noise.'
    )

    parser.add_argument(
        '-dir2', '--data-dir2',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd1_w3" / "non_augmented",
        help='Path to a directory of ground truth images (without noise).'
    )
    parser.add_argument(
        '-gs', '--grid-search',
        type=bool,
        default=False,
        help='Parameter that controls whether to perform grid search.'
    )

    dir1 = parser.parse_args().data_dir1
    dir2 = parser.parse_args().data_dir2
    gs = parser.parse_args().grid_search
    if not dir1.is_dir():
        raise ValueError(f"{dir1} is not a valid directory.")
    if not dir2.is_dir():
        raise ValueError(f"{dir2} is not a valid directory.")
    
    # Make sure output directory exists
    (SCRIPT_DIR / "outputs").mkdir(exist_ok=True)

    # Read original image (with noise)
    og_images = list(read_images(dir1))

    # Read ground truth images (without noise)
    gt_images = list(read_images(dir2))

    # Compute Grid Search
    if gs:
        # Grid Search over param grid
        best_thr, best_score = grid_search_thresholds(og_images, gt_images, BASE_THRESHOLDS, SEARCH_SPACE)

        print("Best thresholds found:")
        for k, v in best_thr.items():
            print(f"  {k}: {v}")
        print(f"Best mean score: {best_score:.4f}")

        # Save the results to a file
        with open(SCRIPT_DIR / "outputs" / "best_thresholds.txt", "w") as f:
            f.write("Best thresholds found:\n")
            for k, v in best_thr.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"Best mean score: {best_score:.4f}\n")
    else:
        # Denoise images
        denoised = denoise_batch(og_images, thresholds=BEST_THRESHOLDS)

        # Compute score of best implementation
        scores, mean_score = eval_batch(og_images, denoised, gt_images)

        print(f"Mean score: {mean_score:.4f}")

        scores_path = SCRIPT_DIR / "outputs" / "denoising_scores.txt"
        with open(scores_path, "w") as f:
            f.write(f"Mean Score: {mean_score:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write("Individual Image Scores:\n")
            for i, score in enumerate(scores):
                f.write(f"  Image {i}: {score:.4f}\n")
        print(f"Scores saved to {scores_path}")

        plot_path = SCRIPT_DIR / "outputs" / "denoising_comparison.png"
        plot_all_comparisons(
            og_images, 
            denoised, 
            gt_images, 
            scores, 
            save_path=plot_path
        )

        # # Plot results
        # plot_image_comparison(den_images, og_images, 10)