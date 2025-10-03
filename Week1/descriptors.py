from typing import List, Optional, Any, Union, Dict

import numpy as np
from pathlib import Path

from color_spaces import rgb_to_hsv, rgb_to_xyz
from histogram import histogram
from io_utils import write_pickle, read_images

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
    R = rgb[..., 0] / 255.0
    G = rgb[..., 1] / 255.0
    B = rgb[..., 2] / 255.0

    # Use pdf insetad of counts
    pR = histogram(R, n_bins=R_bins, ) / 3
    pG = histogram(G, n_bins=G_bins, ) / 3
    pB = histogram(B, n_bins=B_bins, ) / 3

    return np.concatenate([pR, pG, pB], axis=0).astype(np.float32)


def _desc_hsv(
    rgb: np.ndarray,
    H_bins: int = 64,
    S_bins: int = 16,
    V_bins: int = 16,

    ) -> np.ndarray:
    """
    HSV 1D histogram descriptor.
    By default use H and S. If use_value=True, adds V channel as well.
    """
    # From RGB to HSV
    hsv = rgb_to_hsv(rgb).astype(np.float32)

    def _compute_hist(hsv: np.ndarray) -> np.ndarray:
        H = hsv[..., 0] / 360.0
        S = hsv[..., 1]
        V = hsv[..., 2]

        hH = histogram(H, n_bins=H_bins, )
        hS = histogram(S, n_bins=S_bins, )

        desc_parts = [hH.astype(np.float32) / 2, hS.astype(np.float32) / 2]
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
        quad_descs = [ _compute_hist(q) for q in quadrants]
        return np.concatenate([hsv_baseline] + quad_descs, axis=0)

    return hsv_baseline

def _desc_hs_rgb(
    rgb: np.ndarray,
    H_bins: int = 64,
    S_bins: int = 16,
    R_bins: int = 32,
    G_bins: int = 32,
    B_bins: int = 32,

    ) -> np.ndarray:
    """
    HSV 1D histogram descriptor.
    By default use H and S. If use_value=True, adds V channel as well.
    """
    # From RGB to HSV
    hsv = rgb_to_hsv(rgb).astype(np.float32)

    def _compute_hist(hsv: np.ndarray,rgb: np.ndarray) -> np.ndarray:
        H = hsv[..., 0] / 360
        S = hsv[..., 1]

        R = rgb[..., 0] / 255.0
        G = rgb[..., 1] / 255.0
        B = rgb[..., 2] / 255.0
        hH = histogram(H, n_bins=H_bins, )
        hS = histogram(S, n_bins=S_bins, )
        pR = histogram(R, n_bins=R_bins, )
        pG = histogram(G, n_bins=G_bins, )
        pB = histogram(B, n_bins=B_bins, )


        print(hH.shape,hS.shape,pR.shape,pG.shape,pB.shape)

        desc_parts = [hH.astype(np.float32) / 5, hS.astype(np.float32) / 5, pR.astype(np.float32) / 5, pG.astype(np.float32), pB.astype(np.float32) / 5]

        
        
        return np.concatenate(desc_parts, axis=0)

    hs_rgb_baseline = _compute_hist(hsv,rgb)
    print(hs_rgb_baseline.shape)
    return hs_rgb_baseline


def _desc_hs(
    rgb: np.ndarray,
    H_bins: int = 64,
    S_bins: int = 16,

    ) -> np.ndarray:
    """
    HSV 1D histogram descriptor.
    By default use H and S. If use_value=True, adds V channel as well.
    """
    # From RGB to HSV
    hsv = rgb_to_hsv(rgb).astype(np.float32)

    def _compute_hist(hsv: np.ndarray,rgb: np.ndarray) -> np.ndarray:
        H = hsv[..., 0] / 360.0
        S = hsv[..., 1]

        hH = histogram(H, n_bins=H_bins, )
        hS = histogram(S, n_bins=S_bins, )


        desc_parts = [hH.astype(np.float32), hS.astype(np.float32)]

        
        
        return np.concatenate(desc_parts, axis=0)

    hsv_baseline = _compute_hist(hsv,rgb)


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

    # HSV without quadrants should work best with chi-squared metric
    # HSV with quadrants should work best with chi-squared but weighted ((global 0.5; each 2×2 cell 0.125))
    # If stron illumantion change use Bhattacharyya
    elif method == "hsv":
        H_bins = params.get("H_bins", 32)
        S_bins = params.get("S_bins", 32)
        V_bins = params.get("V_bins", 32)
        use_value = params.get("use_value", False)
        quadrants = params.get("quadrants", False)
        print(f"Computing HSV descriptors with quadrants set to {quadrants} and use_value set as {use_value}")
        descs = [ _desc_hsv(im, H_bins, S_bins, V_bins,) \
                  for im in img_list]
    elif method == "hs_rgb":
        H_bins = params.get("H_bins", 32)
        S_bins = params.get("S_bins", 32)
        V_bins = params.get("V_bins", 32)
        R_bins = params.get("R_bins", 32)
        G_bins = params.get("G_bins", 32)
        B_bins = params.get("B_bins", 32)
        use_value = params.get("use_value", False)
        quadrants = params.get("quadrants", False)
        print(f"Computing HS_RGB descriptors with quadrants set to {quadrants} and use_value set as {use_value}")
        descs = [ _desc_hs_rgb(im, H_bins, S_bins, R_bins, G_bins, B_bins, ) \
                  for im in img_list]
    elif method == "hs":
        H_bins = params.get("H_bins", 32)
        S_bins = params.get("S_bins", 32)
        use_value = params.get("use_value", False)
        quadrants = params.get("quadrants", False)
        print(f"Computing HS descriptors with quadrants set to {quadrants} and use_value set as {use_value}")
        descs = [ _desc_hs(im, H_bins, S_bins,) \
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
    imgs = read_images(Path.cwd() / "BBDD")

    bbdd_desc = compute_descriptors(imgs, method="hs_rgb",
                                    params=dict(R_bins=32, G_bins=32, B_bins=32),
                                    save_path= Path.cwd()/ "BBDD_2" / "BBDD_2_descriptors_hs_rgb.pkl")

    # Set quadrants and use_value as wished
    # hsv_desc = compute_descriptors(imgs, method = "hsv",
    #                                params=dict(H_bins=64, S_bins=16, V_bins=16,
    #                                            use_value=False, quadrants = True),
    #                                 save_path=None)