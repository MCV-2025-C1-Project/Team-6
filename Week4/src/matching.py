from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from descriptors import compute_one_descriptor, ORBExtractor, SIFTExtractor, HarrisSIFTExtractor
from params import FLANN_MATCHER_PARAMS
"""
information extracted from: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
"""


### Helpers ###
# This helpers let us define the matcher backend to use: brute force or flann.
def _norm_for(descriptor) -> int:
    return cv.NORM_L2 if descriptor == "sift" else cv.NORM_HAMMING

def _bf(descriptor, cross_check: bool) -> cv.BFMatcher:
    return cv.BFMatcher(normType=_norm_for(descriptor), crossCheck=cross_check)

def _flann(descriptor) -> cv.FlannBasedMatcher:
    return cv.FlannBasedMatcher(FLANN_MATCHER_PARAMS["index_params"][f"{descriptor}"], 
                                FLANN_MATCHER_PARAMS["SearchParams"])

def _get_matcher(descriptor, backend, cross_check: bool = False):
    return _bf(descriptor, cross_check) if backend == "bf" else _flann(descriptor)


### Main method ###
def crosscheck_matches(
    des1: np.ndarray, 
    des2: np.ndarray,
    desc = "sift",
    top_k: Optional[int] = None) -> List[cv.DMatch]:
    """
    BF cross-check (fast). Does not apply ratio, less discriminative, more false positives.
    """
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("No descriptors to match.")
        return []
 
    # Cross-check only makes sense with BF; if backend=="flann", fall back to BF.
    matcher = _bf(desc, cross_check=True)
    matches = matcher.match(des1, des2)
    matches.sort(key=lambda m: m.distance)
    return matches[:top_k] if top_k is not None else matches

def ratio_matches(
    des1: np.ndarray, 
    des2: np.ndarray,
    desc = "sift", 
    backend = "bf",
    ratio: float = 0.75, 
    top_k: Optional[int] = None) -> List[cv.DMatch]:
    """
    KNN + Lowe's ratio. Works with BF or FLANN.
    """
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("No descriptors to match.")
        return []
    
    matcher = _get_matcher(desc, backend, cross_check=False)
    knn = matcher.knnMatch(des1, des2, k=2) # 2 neighbors for the ratio test

    good = []
    for neigh in knn:
        if len(neigh) < 2: continue
        m, n = neigh[0], neigh[1] # m best match, n second best
        if m.distance < ratio * n.distance:
            good.append(m) # match only trustworthy if best neighbor significantly closer than second-best one.
    good.sort(key=lambda m: m.distance)
    return good[:top_k] if top_k is not None else good

def bidirectional_ratio_matches(
    des1: np.ndarray,
    des2: np.ndarray,
    desc = "sift", 
    backend = "bf",
    ratio: float = 0.75, 
    k: int = 2, 
    top_k: Optional[int] = None) -> List[cv.DMatch]:
    """
    Bidirectional ratio. queryIdx descriptor of the query, trainIDx descriptor that matches
    in bbdd and distance its distance.
    """
    fwd = ratio_matches(des1, des2, desc, backend, ratio, top_k=None)
    if not fwd: return []
    rev = ratio_matches(des2, des1, desc, backend, ratio, top_k=None)
    rev_pairs = {(m.queryIdx, m.trainIdx) for m in rev}  # (idx in B, idx in A)
    mutual = [m for m in fwd if (m.trainIdx, m.queryIdx) in rev_pairs]
    mutual.sort(key=lambda m: m.distance)
    return mutual[:top_k] if top_k is not None else mutual


if __name__=="__main__":
    SCRIPT_DIR = Path(__file__).resolve().parents[1]
    img1 = cv.imread(SCRIPT_DIR.parent / 'qsd1_w4' / '00001.jpg')
    img2 = cv.imread(SCRIPT_DIR.parent / 'qsd1_w4' / '00002.jpg')
    gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    orb = ORBExtractor()
    sift = SIFTExtractor()
    harris = HarrisSIFTExtractor()

    kp1, des1 = compute_one_descriptor(gray1, method='orb',sift=sift,orb=orb, hsift=harris)
    kp2, des2 = compute_one_descriptor(gray2, method='orb',sift=sift,orb=orb, hsift=harris)
    
    matches = bidirectional_ratio_matches(des1, des2, desc='orb', backend='flann')
    vis = cv.drawMatches(img1, kp1, img2, kp2, matches, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()












