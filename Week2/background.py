from typing import List

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.ndimage import (binary_opening, 
                           binary_closing, 
                           generate_binary_structure,
                           binary_propagation, 
                           binary_fill_holes,
                           median_filter)

from color_spaces import rgb_to_hsv, rgb_to_lab

from scipy.ndimage import rotate as _rotate

def _integral_image(img: np.ndarray) -> np.ndarray:
    ii = img.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    return np.pad(ii, ((1,0),(1,0)), mode='constant')

def _rect_sum(ii: np.ndarray, top: int, left: int, bottom: int, right: int) -> int:
    return ii[bottom, right] - ii[top, right] - ii[bottom, left] + ii[top, left]

def best_centered_rect_mask(mask: np.ndarray,
                            min_h: int = 0, min_w: int = 0,
                            min_frac: float = 0.5,   # ← mínimo relativo (0.5 = mitad)
                            step: int = 4,
                            lambda_penalty: float = 1.0,
                            margin: int = 8) -> np.ndarray:
    """
    Devuelve una máscara booleana con el mejor rectángulo *centrado*.
    Impone tamaño mínimo relativo: alto>=min_frac*H y ancho>=min_frac*W.
    Score = white - lambda_penalty * black.
    """
    assert mask.ndim == 2
    H, W = mask.shape
    cy, cx = H // 2, W // 2

    # mínimos absolutos teniendo en cuenta el mínimo relativo
    min_h = max(min_h, int(np.ceil(min_frac * H)))
    min_w = max(min_w, int(np.ceil(min_frac * W)))

    # evitar que el margen haga imposible el mínimo
    usable_h = H - 2 * margin
    usable_w = W - 2 * margin
    min_h = min(min_h, max(1, usable_h))
    min_w = min(min_w, max(1, usable_w))

    ii = _integral_image(mask)

    # máximos respetando márgenes
    max_h = usable_h
    max_w = usable_w

    best_score = -np.inf
    best_t = best_l = best_b = best_r = 0

    for h in range(min_h, max_h + 1, step):
        half_h = h // 2
        top    = max(cy - half_h, margin)
        bottom = min(top + h, H - margin)
        if bottom - top < min_h:  # por si el centro y el margen lo impiden
            continue

        for w in range(min_w, max_w + 1, step):
            half_w = w // 2
            left   = max(cx - half_w, margin)
            right  = min(left + w, W - margin)
            if right - left < min_w:
                continue

            whites = _rect_sum(ii, top, left, bottom, right)
            area   = (bottom - top) * (right - left)
            blacks = area - whites
            score  = whites - lambda_penalty * blacks

            if score > best_score:
                best_score = score
                best_t, best_l, best_b, best_r = top, left, bottom, right

    rect_mask = np.zeros_like(mask, dtype=bool)
    rect_mask[best_t:best_b, best_l:best_r] = True
    return rect_mask

def best_rotated_mask(original_mask: np.ndarray,
                      rect_mask: np.ndarray,
                      angle_limit: float = 30.0,
                      angle_step: float = 1.0,
                      lambda_penalty: float = 1.2) -> np.ndarray:
    """
    Dado un rectángulo centrado (rect_mask), busca la mejor rotación ±angle_limit
    que maximiza whites - lambda_penalty*blacks respecto a original_mask.

    original_mask : (H,W) bool  -> máscara “ground” (la salida Mahalanobis antes del rectángulo)
    rect_mask     : (H,W) bool  -> máscara del rectángulo centrado
    """
    assert original_mask.shape == rect_mask.shape
    orig = original_mask.astype(bool)
    base = rect_mask.astype(bool)

    best_score = -np.inf
    best = base

    angles = np.arange(-angle_limit, angle_limit + 1e-9, angle_step)
    for ang in angles:
        # rotación alrededor del centro de la imagen; sin cambiar tamaño
        cand = _rotate(base.astype(np.uint8), ang, reshape=False, order=0, mode="constant", cval=0).astype(bool)

        area   = int(cand.sum())
        if area == 0:
            continue
        whites = int(np.logical_and(orig, cand).sum())
        blacks = area - whites
        score  = whites - lambda_penalty * blacks

        if score > best_score:
            best_score = score
            best = cand

    return best

# TODO: Maybe a more robust way to extract border samples? Remove outliers?
def extract_border_samples(img: np.ndarray, border_width: int = 20) -> np.ndarray:
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

    max_samples = 2000
    if len(samples) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = samples[idx]
    return samples  # shape (N, C)


def remove_background(images: List[np.ndarray], border_width: int = 10, color_space = "lab",
                      use_percentile_thresh=True, percentile=99):

    masks = []
    for _ , img in enumerate(images):
        plt.subplot(1, 2, 1)
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

        # Border taken from the samples, 10 pix by default
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
            delta_s = samples - mean
            d2_s = np.einsum('ij,ij->i', delta_s @ cov_inv, delta_s)
            threshold = np.percentile(d2_s, percentile)
        else:
            # More robust...we do not have a perfect Gaussian distribution!
            threshold = chi2.ppf(1 - 0.01, df=2)

        mask_bg = (d2 <= threshold)

        #struct = generate_binary_structure(2, 1)  # 4-connected (safer than 8) // Like a cross
        struct = np.ones((5, 5), dtype=bool)   # 5x5 (cuadrado)
        candidate_bg = binary_opening(mask_bg, structure=struct, iterations=1)
        candidate_bg = binary_closing(candidate_bg, structure=struct, iterations=2)

        # Propagate background only from border
        # TODO: should we code this from scratch?
        seed = np.zeros_like(candidate_bg, dtype=bool)
        bw = border_width
        seed[:bw, :] = True; seed[-bw:, :] = True
        seed[:, :bw] = True; seed[:, -bw:] = True

        background = binary_propagation(seed, mask=candidate_bg)

        foreground = ~background
        org_mask = binary_fill_holes(foreground)                 # light cleanup
        # Optional light smoothing:
        # foreground = binary_closing(fg, structure=struct, iterations=1)

        rect_mask = best_centered_rect_mask(foreground, min_frac=0.5, step=4, lambda_penalty=1.2)
        foreground = best_rotated_mask(org_mask, rect_mask, angle_limit=30, angle_step= 1, lambda_penalty=1.2)

        plt.subplot(1, 2, 2)
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