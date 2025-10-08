from typing import List, Dict, Any

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.ndimage import (binary_opening, 
                           binary_closing, 
                           binary_propagation, 
                           binary_fill_holes,
                           median_filter)

from scipy.ndimage import rotate as _rotate

from color_spaces import rgb_to_hsv, rgb_to_lab
from params import background_experiments
from metrics import f1_score, precision, recall, intersection_over_union

# we'll use it in remove_background, avoid reallocating it every time
STRUCT = np.ones((5, 5), dtype=bool) 

def _integral_image(img: np.ndarray) -> np.ndarray:
    ii = img.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    return np.pad(ii, ((1,0),(1,0)), mode='constant')

def _rect_sum(ii: np.ndarray, top: int, left: int, bottom: int, right: int) -> int:
    return ii[bottom, right] - ii[top, right] - ii[bottom, left] + ii[top, left]

def best_centered_rect_mask(mask: np.ndarray,
                            min_h: int = 0, 
                            min_w: int = 0,
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

def extract_border_samples(img: np.ndarray, 
                           border_width: int = 20, 
                           max_samples = 2000) -> np.ndarray:
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

    if len(samples) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = samples[idx]
    return samples  # shape (N, C)

def remove_background(img: np.ndarray, 
                      method: dict):

    # Locally smooth the color noise (so we get better results with the masks near the borders!)
    img_smooth = median_filter(img, size=(3, 3, 1))

    # Try a*, b* (LAB) or HS (HSV)
    if method["color_space"] == "lab":
        lab = rgb_to_lab(img_smooth)
        channels = lab[..., 1:3]
    
    else:
        hsv = rgb_to_hsv(img_smooth)
        channels = hsv[..., :2] 

        # Normalization (Hue from [0, 360])
        channels[..., 0] /= 360

    # Border taken from the samples, 10 pix by default
    samples = extract_border_samples(channels, border_width=method["border_width"])

    # Minimum Covariance Determinant -> remove a fraction of outliers (more robust than simple covariance)
    cov_fraction = method["cov_fraction"]
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

    if method["use_percentile_thresh"]:
        # More robust...we do not have a perfect Gaussian distribution!
        delta_s = samples - mean
        d2_s = np.einsum('ij,ij->i', delta_s @ cov_inv, delta_s)
        threshold = np.percentile(d2_s, method["percentile"])
    else:
        threshold = chi2.ppf(1 - float(1-method["percentile"]), df=2)

    mask_bg = (d2 <= threshold)

    #struct = generate_binary_structure(2, 1)  # 4-connected (safer than 8) // Like a cross
    candidate_bg = binary_opening(mask_bg, structure=STRUCT, iterations=1)
    candidate_bg = binary_closing(candidate_bg, structure=STRUCT, iterations=2)

    # Propagate background only from border
    # TODO: should we code this from scratch?
    seed = np.zeros_like(candidate_bg, dtype=bool)
    bw = method["border_width"]
    # Take the edges as the seed
    seed[:bw, :] = True; seed[-bw:, :] = True
    seed[:, :bw] = True; seed[:, -bw:] = True

    background = binary_propagation(seed, mask=candidate_bg)

    foreground = ~background
    org_mask = binary_fill_holes(foreground) # light cleanup

    rect_mask = best_centered_rect_mask(foreground, min_frac=method["min_frac"], step=method["step"], lambda_penalty=method["lambda_penalty"])
    foreground = best_rotated_mask(org_mask, rect_mask, angle_limit=method["angle_limit"], angle_step=method["angle_step"], lambda_penalty=method["lambda_penalty"])

    return foreground.astype(bool)

def _to_bool_mask(arr: np.ndarray) -> np.ndarray:
    """
    Convert any array to H,W binary mask.
    """
    if arr.ndim == 3:
        arr = arr[..., 0]
    # binarize: treat >127 (or >0) as foreground
    if arr.dtype == bool:
        return arr
    return (arr > 127)

def _evaluate_method(images: List[np.ndarray], gts: List[np.ndarray], desc: Dict[str, Any]):
    n = min(len(images), len(gts))
    preds = [remove_background(images[i], desc).astype(bool) for i in range(n)]
    preds = [_to_bool_mask(p) for p in preds]
    gts   = [_to_bool_mask(gts[i]) for i in range(n)]

    ious = [intersection_over_union(gt, pr) for gt, pr in zip(gts, preds)]
    f1s  = [f1_score(gt, pr)                for gt, pr in zip(gts, preds)]
    pres = [precision(gt, pr)               for gt, pr in zip(gts, preds)]
    recs = [recall(gt, pr)                  for gt, pr in zip(gts, preds)]

    scores = {
        "iou": float(np.mean(ious)) if ious else 0.0,
        "f1":  float(np.mean(f1s))  if f1s  else 0.0,
        "precision": float(np.mean(pres)) if pres else 0.0,
        "recall":    float(np.mean(recs)) if recs else 0.0,
    }
    return scores, preds

def find_best_mask(images: List[np.ndarray], masks_gt: List[np.ndarray]) -> Dict[str, Any]:
    best = None
    results = []

    for desc in background_experiments:
        print(f"Using method: {desc}")
        scores, _ = _evaluate_method(images, masks_gt, desc)
        results.append({"method": desc, "scores": scores})
        if (best is None) or (scores["iou"] > best["scores"]["iou"]) or \
           (np.isclose(scores["iou"], best["scores"]["iou"]) and scores["f1"] > best["scores"]["f1"]):
            best = {"method": desc, "scores": scores}
        print(f"Score: {best['scores']}")

    with open("./Week2/background_best_results.txt", "w") as f:
        f.write("Best method:\n")
        for k, v in best["method"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nBest scores:\n")
        for k, v in best["scores"].items():
            f.write(f"  {k}: {v:.4f}\n")

    return {
        "best_method": best["method"],
        "best_scores": best["scores"]
    }

def apply_best_method_and_plot(
    images: List[np.ndarray],
    best_method: Dict[str, Any]):
    """
    Applies the best method and plots Original vs Mask, returns masks.
    """
    masks = [remove_background(img, best_method).astype(bool) for img in images]
    n = len(masks)

    for i in range(n):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1); plt.imshow(images[i]); plt.title("Original"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(masks[i], cmap="gray"); plt.title("Mask"); plt.axis("off")
        plt.tight_layout(); plt.show(); plt.close()

    return masks
    
# TODO: With the proposed posible method, all  results are the same! ,maybe we need to change some other 
# params. also at the end we'll need just to apply best method to the images, refactor this.
# Finally, could perform a real grid search, now we just try options