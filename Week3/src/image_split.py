import numpy as np
import cv2
from background_team2 import convert_to_representation, compute_edge_mask

def split_image(im, min_area_ratio=0.2, radius=10, gradient_threshold=0.15,
                min_split_frac=0.30, max_split_frac=0.70):
    """
    Divide 'im' en 2 imágenes usando la separación horizontal entre
    los 2 mayores componentes detectados con compute_edge_mask().
    - El corte vertical se restringe a [min_split_frac, max_split_frac] del ancho.
    """
    # 1) Máscara de bordes -> binaria
    im_lab = convert_to_representation(im)
    mask_bool, _ = compute_edge_mask(im_lab, gradient_threshold=gradient_threshold)
    bin_img = (mask_bool.astype(np.uint8) > 0).astype(np.uint8)

    # 2) Conectar por radio (igual que en tu pipeline)
    def _connected_by_radius(img, radius=3):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 10*radius+1))
        dil = cv2.dilate(img, k, iterations=1)
        num, labels, stats, cents = cv2.connectedComponentsWithStats(dil, connectivity=8)

        labels_orig = np.zeros_like(labels)
        for lab in range(1, num):
            labels_orig[(labels == lab) & (img > 0)] = lab
        return num, labels_orig, stats, cents

    num_labels, labels, stats, _ = _connected_by_radius(bin_img, radius=radius)
    h, w = bin_img.shape
    if num_labels <= 1:
        return [im]

    total = h * w
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    valid_idxs = [i for i in order if areas[i] >= min_area_ratio * total][:2]
    if len(valid_idxs) < 2:
        return [im]

    comp1 = (labels == (valid_idxs[0] + 1))
    comp2 = (labels == (valid_idxs[1] + 1))

    def _bbox_and_center(comp):
        ys, xs = np.where(comp)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        return (x_min, y_min, x_max, y_max), (cx, cy)

    (x1_min, _, x1_max, _), (cx1, _) = _bbox_and_center(comp1)
    (x2_min, _, x2_max, _), (cx2, _) = _bbox_and_center(comp2)

    if cx2 < cx1:
        (x1_min, x1_max, cx1), (x2_min, x2_max, cx2) = (x2_min, x2_max, cx2), (x1_min, x1_max, cx1)

    # 4) Calcular corte propuesto
    if x2_min > x1_max:
        cut_x = int((x1_max + x2_min) * 0.5)
    else:
        cut_x = int(round((cx1 + cx2) * 0.5))

    # --- NUEVO: restringir el corte al rango [min_split_frac, max_split_frac] ---
    # normalizar/asegurar orden
    if max_split_frac < min_split_frac:
        min_split_frac, max_split_frac = max_split_frac, min_split_frac

    lo = max(1, int(np.floor(w * float(min_split_frac))))
    hi = min(w - 1, int(np.ceil(w * float(max_split_frac))))
    if lo >= hi:
        # rango degenerado: usa el centro
        cut_x = w // 2
    else:
        cut_x = min(max(cut_x, lo), hi)

    # 5) Cortar imagen
    left_img  = im[:, :cut_x].copy()
    right_img = im[:, cut_x:].copy()
    return [left_img, right_img]

   