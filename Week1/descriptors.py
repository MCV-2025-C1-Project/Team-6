from typing import List, Optional, Any, Union, Dict

import numpy as np
from pathlib import Path

from color_spaces import rgb_to_hsv
from histogram import histogram
from io_utils import write_pickle, read_images

    
# TODO: Create the descriptors. It can be extended and can be done by using histogram.py

# Helpers

def _desc_rgb(
    rgb: np.ndarray,
    R_bins: int = 32,
    G_bins: int = 32,
    B_bins: int = 32
    ) -> np.ndarray:
    """
    RGB 1D histogram per channel.
    """
    rgb = rgb.astype(np.float32)

    # Flatten the channels
    R = rgb[..., 0].ravel()
    G = rgb[..., 1].ravel()
    B = rgb[..., 2].ravel()

    # Use pdf insetad of counts
    _ , pR, _ = histogram(R, n_bins=R_bins, data_range = (0,255), probs=True)
    _ , pG, _ = histogram(G, n_bins=G_bins, data_range = (0,255), probs=True)
    _ , pB, _ = histogram(B, n_bins=B_bins, data_range = (0,255), probs=True)
    
    return np.concatenate([pR, pG, pB], axis=0).astype(np.float32)


def _desc_hsv(
    rgb: np.ndarray,
    H_bins: int = 64,
    S_bins: int = 16,
    V_bins: int = 16,
    use_value: bool = False,
    quadrants: bool = False
    ) -> np.ndarray:
    """
    HSV 1D histogram descriptor.
    By default use H and S. If use_value=True, adds V channel as well.
    """
    # From BGR to HSV
    hsv = rgb_to_hsv(rgb).astype(np.float32)

    def _compute_hist(hsv: np.ndarray) -> np.ndarray:
        H = hsv[..., 0].ravel()
        S = hsv[..., 1].ravel()
        V = hsv[..., 2].ravel()

        _, hH, _ = histogram(H, n_bins=H_bins, data_range=(0, 255), probs=True)
        _, hS, _ = histogram(S, n_bins=S_bins, data_range=(0, 255), probs=True)

        desc_parts = [hH.astype(np.float32), hS.astype(np.float32)]

        if use_value:
            _, hV, _ = histogram(V, n_bins=V_bins, data_range=(0, 255), probs=True)
            desc_parts.append(hV.astype(np.float32))
        
        return np.concatenate(desc_parts, axis=0)

    hsv_baseline = _compute_hist(hsv)

    if quadrants:
        # We divide image in 4 quadrants and create an histogram of each of those
        H, W = hsv.shape[:2]
        h2, w2 = H // 2, W // 2
        quadrants = [
            hsv[0:h2, 0:w2],
            hsv[0:h2, w2:W],
            hsv[h2:H, 0:w2],
            hsv[h2:H, w2:W]
        ]
        desc_quads = np.stack([_compute_hist(q) for q in quadrants], axis=0)
        # Take the mean of the quadrants
        desc_local = np.mean(desc_quads, axis=0)

        # Weighted combination. Mostly single image but also quadrants
        # This way we introduce spatial features!
        return np.concatenate([0.6 * hsv_baseline, 0.4 * desc_local], axis=0)

    return hsv_baseline

def compute_descriptors(imgs: Union[np.ndarray, List[np.ndarray]], 
                        method: str = "rgb",
                        params: Optional[Dict[str, Any]] = None,
                        save_path: Optional[Path] = None) -> List[np.ndarray]:

    img_list: List[np.ndarray] = [imgs] if isinstance(imgs, np.ndarray) else list(imgs)
    params = params or {}

    # Baseline
    if method == "rgb":
        print("Computing RGB descriptors.")
        R_bins = params.get("R_bins", 32)
        G_bins = params.get("G_bins", 32)
        B_bins = params.get("B_bins", 32)
        descs = [ _desc_rgb(im, R_bins, G_bins, B_bins) for im in img_list]
    
    # Much stronger, H and S capture color indpendent of brightness
    elif method == "hsv":
        H_bins = params.get("H_bins", 32)
        S_bins = params.get("S_bins", 32)
        V_bins = params.get("V_bins", 32)
        use_value = params.get("use_value", False)
        quadrants = params.get("quadrants", False)
        print(f"Computing HSV descriptors with quadrants set to {quadrants} and use_value set as {use_value}")
        descs = [ _desc_hsv(im, H_bins, S_bins, V_bins, use_value =use_value, quadrants=quadrants) \
                  for im in img_list]

    else:
        raise ValueError(f"Invalid method ({method}) for computing image descriptors!")
    
    out = np.stack(descs, axis=0)
    if save_path is not None:
        payload = {"method": method, "params": params, "descriptors": out}
        write_pickle(payload, save_path)
        return None 
    else:
        return out
        
    
if __name__=="__main__":
    imgs = read_images(Path.cwd() / "BBDD_2")

    bbdd_desc = compute_descriptors(imgs, method="rgb",
                                    params=dict(R_bins=32, G_bins=32, B_bins=32),
                                    save_path= Path.cwd()/ "BBDD_2" / "BBDD_2_descriptors_rgb.pkl")
    
    # Set quadrants and use_value as wished
    # hsv_desc = compute_descriptors(imgs, method = "hsv",
    #                                params=dict(H_bins=64, S_bins=16, V_bins=16,
    #                                            use_value=False, quadrants = True),
    #                                 save_path=None)