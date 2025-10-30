import numpy as np
import cv2
from typing import List, Tuple, Any

from background_remover import convert_to_representation, compute_edge_mask

def split_image(
    im: np.ndarray,
    min_area_ratio: float = 0.2,
    radius: int = 10,
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
    im_lab = convert_to_representation(im)
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

        # Create a large, vertically-oriented structuring element
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 10*radius+1))
        dil = cv2.dilate(img, k, iterations=1)
        
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

    # Get areas for all labels *except* background (label 0) and sort their indices
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)

    # Filter for components that meet the minimum area ratio
    valid_idxs = [i for i in order if areas[i] >= min_area_ratio * total][:2]

    # If we don't have two valid components, we can't split
    if len(valid_idxs) < 2:
        return [im]

    # Create masks for the two largest components
    comp1 = (labels == (valid_idxs[0] + 1))
    comp2 = (labels == (valid_idxs[1] + 1))


    def _bbox_and_center(comp: np.ndarray) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
        """Calculates the bounding box and center of mass for a component mask."""
        ys, xs = np.where(comp)
        if len(xs) == 0: # Handle empty component just in case
             return (0, 0, 0, 0), (0.0, 0.0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        # Use center of bounding box, not center of mass
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        return (x_min, y_min, x_max, y_max), (cx, cy)
    

    (x1_min, _, x1_max, _), (cx1, _) = _bbox_and_center(comp1)
    (x2_min, _, x2_max, _), (cx2, _) = _bbox_and_center(comp2)

    # Ensure component 1 is the one on the left
    if cx2 < cx1:
        (x1_min, x1_max, cx1), (x2_min, x2_max, cx2) = (x2_min, x2_max, cx2), (x1_min, x1_max, cx1)

    # Calculate proposed split location
    if x2_min > x1_max:
        cut_x = int((x1_max + x2_min) * 0.5)
    else:
        cut_x = int(round((cx1 + cx2) * 0.5))

    # Constrain the split location [min_split_frac, max_split_frac]
    # Ensure min_split_frac is less than max_split_frac
    if max_split_frac < min_split_frac:
        min_split_frac, max_split_frac = max_split_frac, min_split_frac

    # Calculate absolute pixel boundaries for the cut
    lo = max(1, int(np.floor(w * float(min_split_frac))))
    hi = min(w - 1, int(np.ceil(w * float(max_split_frac))))

    if lo >= hi:
        # rango degenerado: usa el centro
        cut_x = w // 2
    else:
        cut_x = min(max(cut_x, lo), hi)

    # Split the image and return
    left_img  = im[:, :cut_x].copy()
    right_img = im[:, cut_x:].copy()
    
    return [left_img, right_img]

   