"""This script removes noise from pictures by using filters """
import argparse
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import uniform_filter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.io_utils import read_images
from utils.color_spaces import rgb_to_ycrcb

SCRIPT_DIR = Path(__file__).resolve().parent
THRESHOLDS  = {
        "sp_impulse": 0.06,   
        "sig_gauss":  0.05,   
        "blk_jpeg":   1.30,   
        "chr_chroma": 0.25,   
        "vol_blur":   80.0    
    }

SEARCH_SPACE = {
    "sp_impulse": [0.06, 0.08, 0.10],
    "sig_gauss":  [0.05, 0.06, 0.07],
    "blk_jpeg":   [1.30, 1.35, 1.40],
    "chr_chroma": [0.25, 0.30, 0.35],
    "vol_blur": [80.0]
}


### Helpers ###
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

# Util to plot two images (not used)
def _vis_compat(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make two images stackable: same size, 3 channels, same dtype."""
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)

    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if b.ndim == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)

    if a.dtype != b.dtype:
        b = b.astype(a.dtype)
    return a, b

# Filters
def mean_filter(image: np.ndarray, kernel_size: tuple = (3, 3, 1)) -> np.ndarray:
    """
    Replaces each pixel with the average of its neighbors.
    Reduces uncorrelated Gaussian noise. Blurs edges and small details and
    Is not robust to outliers.
    """
    return uniform_filter(image, size=kernel_size)

def median_filter(image: np.ndarray, kernel_size: int = 3) ->  np.ndarray:
    """
    Take the median of the pixels under the kernel.
    Non-linear -> it replaces outliers but keeps step edges sharp.
    """
    return cv2.medianBlur(image, kernel_size)

def gaussian_blur(image: np.ndarray, kernel_size: tuple = (15, 15)) ->  np.ndarray:
    """
    Smooths random Gaussian noise while preserving edges a bit better than uniform averaging.
    Still blurs edges and not robust to outliers.
    """
    return cv2.GaussianBlur(image, kernel_size , 0)

def bilateral_filter(image_luminance, d=5, sigma_c=13, sigma_s=3):
    """
    Smooths within regions but doesn’t cross strong edges.
    Used for Gaussian-ish noise or JPEG artifacts
    """
    if image_luminance.dtype != np.uint8:
        image_luminance = np.clip(np.rint(image_luminance), 0, 255).astype(np.uint8)
    out = cv2.bilateralFilter(image_luminance, d=d, sigmaColor=sigma_c, sigmaSpace=sigma_s)
    assert out.shape == image_luminance.shape
    return out
    
def adaptive_median_filter(image: np.ndarray, Smax: int = 7) -> np.ndarray:
    """
    Adaptive Median Filter
    """
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

### Ratios to check noise ###
def salt_and_pepper_ratio(u8_gray_image: np.ndarray,
                          t_extreme: int = 12, # how far we count as 'extreme' in tails
                          win: int = 3, # local window (3 or 5) for median
                          delta_abs: int = 30, # absolute jump to call an impulse
                          k_mad: float = 3.5, # robust (MAD-based) threshold
                          flat_pct: float = 60.0,
                          border: int = 8) -> float:
    """
    Fraction of pixels that behave like isolated impulses.
    Combines (A) near-extreme tails and (B) local outliers vs. local median,
    evaluated mostly on flat regions (to avoid mistaking texture for impulses).
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

def jpeg_blockiness(u8_gray_image: np.ndarray) -> float:
    """
    Low-quality JPEG creates visible jumps exactly at 8×8 block borders.
    If the borders have much larger jumps than the interior, the ratio > 1
    """
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
    if interior < 1e-2:   # scene too flat to judge
        return 1.0
    return float((v_bound + h_bound) / (interior + 1e-6))

def flat_region_sigma(gray_u8: np.ndarray, flat_pct: float = 15.0, var_win: int = 7) -> float:
    """
    Robust estimate of Gaussian-like noise, measured on truly flat pixels.
    'flat_pct' is the percentile of flattest pixels kept (stricter than 30%).
    """
    g = gray_u8.astype(np.float32)

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
    """
    gray_u8 = _to_gray_u8(img)
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F, ksize=ksize)
    return float(lap.var())

def chroma_noise_ratio(rgb: np.ndarray) -> float:
    """
    Simple high-pass residual for each chroma channels.
    Many cameras produce color speckle: fine, grainy noise mostly in chroma
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


### Main methods ###
def measure_noise_metrics(img: np.ndarray) -> dict:
    """
    Compute core metrics used for blind noise detection.
    Returns a dict with:
      sp   : salt-and-pepper ratio (0..1)
      sig  : flat-region sigma (in [0,1] intensity units)
      blk  : JPEG blockiness ratio (~1 clean, >1.1 blocky)
      vol  : variance of Laplacian (sharpness proxy)
      chr  : chroma noise ratio (only for color images; None otherwise)
    """
    gray_u8 = _to_gray_u8(img)
    sp  = salt_and_pepper_ratio(gray_u8)
    sig = flat_region_sigma(gray_u8)
    blk = jpeg_blockiness(gray_u8)
    vol = variance_of_laplacian(img)
    chrn = chroma_noise_ratio(img) if (img.ndim == 3 and img.shape[2] >= 3) else None

    return {"sp": sp, "sig": sig, "blk": blk, "vol": vol, "chr": chrn}

def check_noise(img: np.ndarray,
                thresholds: dict | None = None,
                margin = 0.002) -> list[str]:
    """
    Classify noise types present in the image.
    Returns a list like: ["impulse", "gaussian_like", "jpeg_blockiness", "chroma_noise"]
    If nothing triggers, returns ["clean"].
    """
    # default
    thr = {
        "sp_impulse": 0.015,   # ≥1.5% extreme pixels → impulses likely
        "sig_gauss":  0.016,   # flat-region sigma in [0,1] units
        "blk_jpeg":   1.15,    # >1.15 → visible blocking
        "chr_chroma": 0.25,    # >0.25 → noticeable chroma speckle
        "vol_blur":   80.0     # <80 (8-bit ~1MP) → quite blurry
    }
    if thresholds:
        thr.update(thresholds)

    m = measure_noise_metrics(img)
    labels: list[str] = []

    # Impulse noise (salt & pepper)
    if (m["sp"] >= thr["sp_impulse"] + margin):
        labels.append("impulse")

    # Gaussian-like / random noise
    if m["sig"] >= thr["sig_gauss"] + margin:
        labels.append("gaussian_like")

    # JPEG blocking artifacts
    if (m["blk"] >= thr["blk_jpeg"]) and (m["sig"] >= 0.03 or (m["chr"] or 0) >= 0.10):
        labels.append("jpeg_blockiness")

    # Chroma speckle (only if color metric available)
    if (m["chr"] is not None) and (m["chr"] >= thr["chr_chroma"] + margin):
        labels.append("chroma_noise")

    if not labels:
        labels = ["clean"]

    return labels

def denoise_images(image, thresholds = None, return_labels=False):
    labels = check_noise(image, thresholds)
    if labels == ["clean"]:
        return (image, labels) if return_labels else image
    
    # gray-case
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        out = image.copy()
        if "impulse" in labels:
            out = adaptive_median_filter(out, Smax=7)
            out = median_filter(out, kernel_size=3)
            # polish if also gaussian/jpg flagged
            if ("gaussian_like" in labels) or ("jpeg_blockiness" in labels):
                out = bilateral_filter(out, d=5, sigma_c=14, sigma_s=3)
        elif ("gaussian_like" in labels) or ("jpeg_blockiness" in labels):
            out = bilateral_filter(out, d=5, sigma_c=12, sigma_s=3)
        else:
            out = image  # clean
        return (out, labels) if return_labels else out
    
    # 3 channel case
    ycrcb = rgb_to_ycrcb(image)
    Y, Cr, Cb = cv2.split(ycrcb)
    outY, outCr, outCb = Y.copy(), Cr.copy(), Cb.copy()
    
    if "impulse" in labels:
        outY = adaptive_median_filter(outY, Smax=7)

    if ("gaussian_like" in labels) or ("jpeg_blockiness" in labels):
        outY = bilateral_filter(outY, d=5, sigma_c=12, sigma_s=3)
        outY = median_filter(outY, kernel_size=3)  # remove residual dots
        # smooth chroma a bit (impulse can show up as colored specks)
        outCr = bilateral_filter(outCr, d=7, sigma_c=18, sigma_s=4)
        outCb = bilateral_filter(outCb, d=7, sigma_c=18, sigma_s=4)

    if "chroma_noise" in labels:
        outCr = bilateral_filter(outCr, d=7, sigma_c=15, sigma_s=4)
        outCb = bilateral_filter(outCb, d=7, sigma_c=15, sigma_s=4)

    out = cv2.merge([outY, outCr, outCb])
    out = cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)

    return (out, labels) if return_labels else out

def denoise_batch(images, thresholds=None, return_labels=False):
    outs = []
    for img in tqdm(images, desc="Denoising images..."):
        den = denoise_images(img, thresholds=thresholds, return_labels=return_labels)
        outs.append(den)
    return outs

### Evaluation methods ###
def eval_psnr_ssim(orig, den, gt):
    orig, den, gt = _to_gray_u8(orig), _to_gray_u8(den), _to_gray_u8(gt)

    H,W = gt.shape[:2]
    if orig.shape[:2] != (H,W): orig = cv2.resize(orig, (W,H), cv2.INTER_AREA)
    if den.shape [:2] != (H,W): den  = cv2.resize(den , (W,H), cv2.INTER_AREA)

    if np.array_equal(orig, gt):
        psnr_og = np.inf
        ssim_og = 1.0
    else:
        psnr_og = psnr(gt, orig, data_range=255)
        ssim_og = ssim(gt, orig, data_range=255)

    if np.array_equal(den, gt):
        psnr_den = np.inf
        ssim_den = 1.0
    else:
        psnr_den = psnr(gt, den, data_range=255)
        ssim_den = ssim(gt, den, data_range=255)


    d_psnr = psnr_den - psnr_og
    d_ssim = ssim_den - ssim_og
    eps_psnr, eps_ssim = 0.05, 0.001

    score = max(0.0, d_ssim - eps_ssim) - max(0.0, -(d_ssim) - eps_ssim)
    score += 0.05 * (max(0.0, d_psnr - eps_psnr) - max(0.0, -(d_psnr) - eps_psnr))

    return score

def eval_batch(og_images, den_images, gts):
    scores = []
    for o, d, g in zip(og_images, den_images, gts):
        s = eval_psnr_ssim(o, d, g)
        scores.append(float(s))
    mean_score = float(np.mean(scores)) if scores else 0.0
    return scores, mean_score

def grid_search_thresholds(og, gts, base_thr, search_space):
    best_thr = base_thr.copy()
    best_score = -1e9

    keys = ["sp_impulse", "sig_gauss", "blk_jpeg", "chr_chroma", "vol_blur"]

    for vals in product(*(search_space[k] for k in keys)):
        thr = base_thr.copy()
        for k, v in zip(keys, vals):
            thr[k] = v

        dens = denoise_batch(og, thresholds=thr, return_labels=False)
        _, mean_score = eval_batch(og, dens, gts)

        if mean_score > best_score:
            best_score = mean_score
            best_thr = thr.copy()

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

    dir1 = parser.parse_args().data_dir1
    dir2 = parser.parse_args().data_dir2
    if not dir1.is_dir():
        raise ValueError(f"{dir1} is not a valid directory.")
    if not dir2.is_dir():
        raise ValueError(f"{dir2} is not a valid directory.")
    
    # Read original (w/ noise), ground truths and denoise original
    og_images = list(read_images(dir1))
    gt_images = list(read_images(dir2))
    den_images = denoise_batch(og_images, thresholds=THRESHOLDS)

    # Grid Search over param grid (commented, experiment already done)
    # best_thr, best_score = grid_search_thresholds(og_images, gt_images, THRESHOLDS, SEARCH_SPACE)

    # Compute score of best implementation
    scores, mean_score = eval_batch(og_images, den_images, gt_images)
    # Plot results
    for denoised, original in zip(den_images, og_images):
        img_vis, den_vis = _vis_compat(original, denoised)
        final = np.hstack([img_vis, den_vis])
    
        cv2.imshow('Original vs denoised image', final)
        cv2.waitKey(0)

    cv2.destroyAllWindows()