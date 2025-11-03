import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.io_utils import read_images
import cv2
from typing import List, Tuple, Any
from pathlib import Path
from bg_noise_fliter import denoise_images
from params import BEST_NOISE_PARAMS
from scipy.ndimage import binary_opening


from background_remover import convert_to_representation, compute_edge_mask

SCRIPT_DIR = Path(__file__).resolve().parent
BEST_THRESHOLDS = BEST_NOISE_PARAMS

def split_image(
    im: np.ndarray,
    min_area_ratio: float = 0.1,
    radius: int = 15,
    gradient_threshold: float = 0.15,
    min_split_frac: float = 0.30,
    max_split_frac: float = 0.70
) -> List[np.ndarray]:
    """
    Splits an image into two vertical parts based on component analysis. The vertical
    split is restricted to [min_split_frac, max_split_frac] of width.

    This function attempts to find the two largest connected components in the
    image's edge mask. It then calculates a vertical split point horizontally
    between them. The split location is constrained to lie within a specific
    fractional range of the image's width.

    If two suitable components aren't found, or if the split fails, the
    original image is returned in a list.

    Args:
        im: The input image.
        min_area_ratio: The minimum area a component must have (as a fraction
            of total image area) to be considered one of the two largest.
        radius: The radius used for morphological dilation to connect
            nearby edge components.
        gradient_threshold: The threshold passed to compute_edge_mask to
            detect edges.
        min_split_frac: The minimum allowed horizontal position for the
            vertical cut (as a fraction of image width).
        max_split_frac: The maximum allowed horizontal position for the
            vertical cut (as a fraction of image width).
    Returns:
        A list containing either two image arrays (left and right splits)
        or the single original image array if the split was not possible.
    """

    # Compute edge mask and convert to binary image
    img_den = denoise_images(im, thresholds=BEST_NOISE_PARAMS)
    im_lab = convert_to_representation(img_den)
    mask_bool, _ = compute_edge_mask(im_lab, gradient_threshold=gradient_threshold)
    bin_img = (mask_bool.astype(np.uint8) > 0).astype(np.uint8)

    # Connect components using morphological operations

    def _connected_by_radius(img: np.ndarray, radius: int = 3) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Finds connected components in a binary image after dilation.
        
        This dilates the image to connect nearby components and then finds
        the connected components on the *original* binary image pixels that
        fall under the dilated labels.
        """

         # Elegir elemento estructurante según orientación de la imagen
        h, w = img.shape[:2]
        if w >= h:
            ksize = (2*radius + 1, 10*radius + 1)   # vertical (como estaba)
        else:
            ksize = (10*radius + 1, 2*radius + 1)   # horizontal si la imagen es más alta que ancha
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)

        dil = cv2.dilate(img, k, iterations=1)

        # === ver la máscara aquí mismo: antes y después ===
        # plt.figure(figsize=(10,5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(img, cmap='gray')
        # plt.title("Before dilation")
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.imshow(dil, cmap='gray')
        # plt.title("After dilation")
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()
        # ================================================
        
        # Find components in the dilated image
        num, labels, stats, cents = cv2.connectedComponentsWithStats(dil, connectivity=8)

        # Map labels back to the original image's pixels
        labels_orig = np.zeros_like(labels)
        for lab in range(1, num): # Skip background label 0
            labels_orig[(labels == lab) & (img > 0)] = lab
            
        return num, labels_orig, stats, cents


       
    num_labels, labels, stats, _ = _connected_by_radius(bin_img, radius=radius)
    h, w = bin_img.shape

    # If 0 or 1 components found, can't split
    if num_labels <= 1:
        return [im]

    # Find the two largest components
    total = h * w

    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    valid_idxs = [i for i in order if areas[i] >= min_area_ratio * total][:2]
    if len(valid_idxs) < 2:
        return [im]

    comp1 = (labels == (valid_idxs[0] + 1))
    comp2 = (labels == (valid_idxs[1] + 1))

    def _bbox_and_center(comp: np.ndarray) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
        ys, xs = np.where(comp)
        if len(xs) == 0:
            return (0, 0, 0, 0), (0.0, 0.0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        return (x_min, y_min, x_max, y_max), (cx, cy)

    (x1_min, y1_min, x1_max, y1_max), (cx1, cy1) = _bbox_and_center(comp1)
    (x2_min, y2_min, x2_max, y2_max), (cx2, cy2) = _bbox_and_center(comp2)

    # --- NUEVO: elegir tipo de corte según orientación ---
    if w >= h:
        # === Corte vertical (como estaba) ===
        if cx2 < cx1:
            (x1_min, x1_max, cx1), (x2_min, x2_max, cx2) = (x2_min, x2_max, cx2), (x1_min, x1_max, cx1)

        if x2_min > x1_max:
            cut_x = int((x1_max + x2_min) * 0.5)
        else:
            cut_x = int(round((cx1 + cx2) * 0.5))

        if max_split_frac < min_split_frac:
            min_split_frac, max_split_frac = max_split_frac, min_split_frac
        lo = max(1, int(np.floor(w * float(min_split_frac))))
        hi = min(w - 1, int(np.ceil(w * float(max_split_frac))))
        cut_x = (w // 2) if lo >= hi else min(max(cut_x, lo), hi)

        left_img  = im[:, :cut_x].copy()
        right_img = im[:, cut_x:].copy()
        return [left_img, right_img]
    else:
        # === Corte horizontal (nuevo para orientación vertical) ===
        if cy2 < cy1:
            (y1_min, y1_max, cy1), (y2_min, y2_max, cy2) = (y2_min, y2_max, cy2), (y1_min, y1_max, cy1)

        if y2_min > y1_max:
            cut_y = int((y1_max + y2_min) * 0.5)
        else:
            cut_y = int(round((cy1 + cy2) * 0.5))

        if max_split_frac < min_split_frac:
            min_split_frac, max_split_frac = max_split_frac, min_split_frac
        lo = max(1, int(np.floor(h * float(min_split_frac))))
        hi = min(h - 1, int(np.ceil(h * float(max_split_frac))))
        cut_y = (h // 2) if lo >= hi else min(max(cut_y, lo), hi)

        top_img    = im[:cut_y, :].copy()
        bottom_img = im[cut_y:, :].copy()
        return [top_img, bottom_img]





   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dir1', '--data-dir1',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd1_w4",
        help='Path to a directory of images without background.'
    )

    dir1 = parser.parse_args().data_dir1

    # Check directory
    if not dir1.is_dir():
        raise ValueError(f"{dir1} is not a valid directory.")

    images = read_images(dir1)

    for idx, img in enumerate(images):
        splits = split_image(img)

        if len(splits) == 1:
            # Mostrar solo una imagen
            rgb = cv2.cvtColor(splits[0], cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 6))
            plt.imshow(rgb)
            plt.title(f"Image {idx} (no split)")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            # Mostrar dos imágenes lado a lado
            rgb1 = cv2.cvtColor(splits[0], cv2.COLOR_BGR2RGB)
            rgb2 = cv2.cvtColor(splits[1], cv2.COLOR_BGR2RGB)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(rgb1)
            axes[0].set_title(f"Image {idx} - Left/Top")
            axes[0].axis('off')
            axes[1].imshow(rgb2)
            axes[1].set_title(f"Image {idx} - Right/Bottom")
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
