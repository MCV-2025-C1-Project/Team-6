"""Implementations of color space conversions based on the website: http://brucelindbloom.com/ 
and the website: https://www.rapidtables.com/convert/color/rgb-to-hsv.html"""
import numpy as np

# Helpers
# When we load an image the pixel values we get are not linear with respect 
# to real light intensity (it has gamma curvature), need to convert back to linear light values
def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Inverse sRGB companding. From gamma-encoded to linear."""
    a = 0.055
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + a) / (1 + a)) ** 2.4
    )

def _f_lab(t: np.ndarray) -> np.ndarray:
    """CIE Lab helper f(t). Non-linear transofrmation with delta 6/29."""
    delta = 6/29
    t0 = delta**3
    return np.where(
        t > t0,
        np.cbrt(t),
        t / (3 * delta**2) + 4/29
    )

# Color space conversions
def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    og_shape = rgb.shape
    rgb = rgb.astype(np.float64)

    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    # Inverse sRGB
    rgb_lin = _srgb_to_linear(rgb)

    # sRGB (linear) -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float64)

    XYZ = (rgb_lin.reshape(-1, 3) @ M.T) * 100.0
    return XYZ.reshape(og_shape).astype(np.float32)

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    og_shape = rgb.shape
    rgb = rgb.astype(np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = rgb.reshape(-1, 3)

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

def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """XYZ -> Lab."""
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("xyz_to_lab expects (H, W, 3).")
    
    _Xn, _Yn, _Zn = 95.047, 100.000, 108.883
    X = xyz[..., 0] / _Xn
    Y = xyz[..., 1] / _Yn
    Z = xyz[..., 2] / _Zn

    fX = _f_lab(X)
    fY = _f_lab(Y)
    fZ = _f_lab(Z)

    L = 116.0 * fY - 16.0
    a = 500.0 * (fX - fY)
    b = 200.0 * (fY - fZ)

    return np.stack([L, a, b], axis=-1).astype(np.float32)

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (D65) -> Lab (float32)."""
    xyz = rgb_to_xyz(rgb)      
    return xyz_to_lab(xyz)
