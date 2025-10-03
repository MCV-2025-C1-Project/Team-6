import numpy as np
"""Implementations of color space conversions based on the website: http://brucelindbloom.com/ 
and the website: https://www.rapidtables.com/convert/color/rgb-to-hsv.html"""

def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    og_shape = rgb.shape
    rgb = rgb.reshape(3, -1) / 255.0
    # We assume sRGB, as its not specified
    M = np.array([[0.4124564  , 0.3575761  , 0.1804375],
                  [0.2126729  , 0.7151522  ,0.0721750 ],
                  [0.0193339   , 0.1191920   ,0.9503041  ]])

    XYZ = np.dot(M, rgb)
    XYZ = (XYZ * 100).astype(np.uint8)
    XYZ = XYZ.reshape(og_shape)

    return XYZ

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    og_shape = rgb.shape
    rgb = rgb.reshape(-1, 3) / 255.0

    R = rgb[:, 0]
    G = rgb[:, 1]
    B = rgb[:, 2]
    Cmax = np.max(rgb,axis=1)
    Cmin = np.min(rgb,axis=1)
    delta = Cmax - Cmin

    H = np.zeros_like(Cmax, dtype=np.float32)
    V = Cmax
    S = np.divide(delta, Cmax, out=np.zeros_like(Cmax, dtype=np.float32), where=Cmax != 0)

    # Safe inverse to avoid /0
    inv_delta = np.divide(1.0, delta, out=np.zeros_like(delta, dtype=np.float32), where=delta != 0)

    mask_r = (delta != 0.0) & (Cmax == R)
    mask_g = (delta != 0.0) & (Cmax == G)
    mask_b = (delta != 0.0) & (Cmax == B)

    H = np.where(mask_r, 60 * (((G - B) * inv_delta) % 6), H)
    H = np.where(mask_g, 60 * (((B - R) * inv_delta) + 2), H)
    H = np.where(mask_b, 60 * (((R - G) * inv_delta) + 4), H)
    H = (H + 360) % 360

    HSV = np.stack([H, S, V], axis=-1).astype(np.float32)
    HSV = HSV.reshape(og_shape)
    return HSV

def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr_to_rgb expects an image of shape (H, W, 3).")

    og_shape = bgr.shape
    flat = bgr.reshape(-1, 3) # No need to normalize, just a swapping of channels

    # BGR -> RGB, swap first and last columns
    B = flat[:, 0]
    G = flat[:, 1]
    R = flat[:, 2]

    rgb_flat = np.stack([R, G, B], axis=-1)
    rgb = rgb_flat.reshape(og_shape)
    return rgb