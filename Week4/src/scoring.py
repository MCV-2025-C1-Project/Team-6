import numpy as np
import cv2 as cv
from pathlib import Path
from typing import Tuple

from descriptors import compute_descriptors
from utils.io_utils import read_images
from matching import bidirectional_ratio_matches, ratio_matches, bf, flann

SCRIPT_DIR = Path(__file__).resolve().parent

#OpenCL can be slower/more variable
try:
    cv.ocl.setUseOpenCL(False)
except Exception:
    pass

### Scoring methods ###
# Number of surviving matches
def score_matches_len(des_query, des_target, desc="sift", backend="bf",
                      use_mutual=True, ratio=0.75, matcher_fwd=None, matcher_rev=None) -> int:
    if use_mutual:
        matches = bidirectional_ratio_matches(des_query, 
                                              des_target, 
                                              desc=desc, 
                                              backend=backend, 
                                              ratio=ratio,
                                            matcher_fwd=matcher_fwd, 
                                            matcher_rev=matcher_rev )
    else:
        matches = ratio_matches(des_query, des_target, desc=desc, backend=backend, ratio=ratio,matcher=matcher_fwd)
    return len(matches)

# RANSAC
def score_matches_inliers(kp_query, des_query, kp_target, des_target, desc="sift", backend="bf",
                          use_mutual=True, ratio=0.75, model="homography",
                          ransac_reproj=3.0, matcher_fwd=None, matcher_rev=None) -> Tuple[int, float]:
    # tentative matches
    if use_mutual:
        matches = bidirectional_ratio_matches(des_query, 
                                              des_target, 
                                              desc=desc, 
                                              backend=backend, 
                                              ratio=ratio,
                                               matcher_fwd=matcher_fwd, 
                                               matcher_rev=matcher_rev)
    else:
        matches = ratio_matches(des_query, des_target, desc=desc, backend=backend, ratio=ratio, matcher=matcher_fwd)

    min_needed = 8 if model == "homography" else 4
    if len(matches) < min_needed:
        return 0, 0.0
    
    # Cap matches so ransac is much faster
    matches = matches[:250]

    # geometric verification
    # x, y coordinates arrays of the keypoints
    src = np.array([kp_query[m.queryIdx].pt for m in matches], dtype=np.float32)
    dst = np.array([kp_target[m.trainIdx].pt for m in matches], dtype=np.float32)

    # we might have roation, scaling, partial occlusion, noise... better to 
    # estimate a transformation that maps them
    if model == "homography":
        H, mask = cv.findHomography(src, dst, cv.RANSAC, ransac_reproj)
    else:  # affine
        H, mask = cv.estimateAffinePartial2D(src, dst, method=cv.RANSAC, ransacReprojThreshold=ransac_reproj)

    if H is None or mask is None:
        return 0, 0.0

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / len(matches)
    return inliers, inlier_ratio # will help us know if the matches are posible or not

### Ranking methods ###
def rank_gallery(
    kp_query=None, 
    des_query=None,
    bbdd_kp=None, 
    bbdd_desc=None,
    desc="sift", 
    backend="bf",
    use_mutual=True,
    use_inliers=False,
    model="homography", 
    ransac_reproj=3.0):
    """
    Ranks database images for one query using either:
      - raw match count (#matches)
      - RANSAC inliers (#inliers)
    Returns a list of tuples:
      - if use_inliers: [(img_id, inliers, inlier_ratio), ...]
      - else: [(img_id, matches), ...]
    """
    print("Ranking gallery...")
    ratio = 0.75 if desc in ["sift", "hsift"] else 0.85
    if backend == "bf":
        matcher_fwd = bf(desc, cross_check=False)
        matcher_rev = bf(desc, cross_check=False) if use_mutual else None
    else:  # "flann"
        matcher_fwd = flann(desc)
        matcher_rev = flann(desc) if use_mutual else None

    scores = []
    for i, (kp_t, des_t) in enumerate(zip(bbdd_kp or [None]*len(bbdd_desc), bbdd_desc)):
        if use_inliers:
            # quick filter to see if we should use ransac or not
            if use_mutual:
                prelim = bidirectional_ratio_matches(
                    des_query, des_t,
                    desc=desc, backend=backend, ratio=ratio, top_k=300,
                    matcher_fwd=matcher_fwd,          
                    matcher_rev=matcher_rev           
                )
            else:
                prelim = ratio_matches(
                    des_query, des_t,
                    desc=desc, backend=backend, ratio=ratio, top_k=300,
                    matcher=matcher_fwd              
                )
            if len(prelim) < 20:  # MIN_TENTATIVE
                scores.append((i, 0, 0.0))
                continue

            inl, inl_ratio = score_matches_inliers(
                kp_query, des_query, kp_t, des_t,
                desc=desc, backend=backend, use_mutual=use_mutual,
                ratio=ratio, model=model, ransac_reproj=ransac_reproj,
                matcher_fwd=matcher_fwd, matcher_rev=matcher_rev)
            scores.append((i, inl, inl_ratio))
        else:
            s = score_matches_len(des_query, des_t, desc=desc, backend=backend,
                                  use_mutual=use_mutual, ratio=ratio,matcher_fwd=matcher_fwd, 
                                  matcher_rev=matcher_rev)
            scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores # finds best matching painting

def decide_unknown(
    ranked,
    use_inliers=False,
    T_matches=12,
    T_inl=15,
    T_ratio=0.30,
    margin=0):
    """
    Decide if top match is valid or 'unknown'.
    Returns (is_unknown, best_id or -1, best_score, secondary_score)
    """
    if not ranked:
        return True, -1, 0, 0

    best_id, best = ranked[0][0], ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0 # second -best match's score

    if use_inliers:
        inl_ratio = ranked[0][2] if len(ranked[0]) > 2 else 0
        is_unknown = (best < T_inl) or (inl_ratio < T_ratio) or ((best - second) < margin)
        return is_unknown, (-1 if is_unknown else best_id), best, inl_ratio
    else:
        is_unknown = (best < T_matches) or ((best - second) < margin)
        return is_unknown, (-1 if is_unknown else best_id), best, second
    
def decide_unknown_calibrated(
    ranked,                      
    use_inliers: bool = True,
    T_inl_abs: int = 15,
    T_ratio_abs: float = 0.30,
    T_peak_ratio: float = 1.5,    # b / s
    T_z: float = 3.0,             # (b - mean)/std over top-K
    k_stat: int = 50
):
    if not ranked:
        return True, -1, 0, 0.0

    vals = np.array([r[1] for r in ranked[:k_stat]], dtype=float)
    b = float(vals[0])
    s = float(vals[1]) if len(vals) > 1 else 0.0
    mu = float(vals[1:].mean()) if len(vals) > 2 else 0.0
    sd = float(vals[1:].std(ddof=1)) if len(vals) > 3 else 1.0
    z  = (b - mu) / max(sd, 1e-6)

    if use_inliers:
        # keep inlier ratio guard
        inl_ratio = float(ranked[0][2]) if len(ranked[0]) > 2 else 0.0
        if inl_ratio < T_ratio_abs:
            return True, -1, b, inl_ratio

    if b < T_inl_abs:
        return True, -1, b, (ranked[0][2] if use_inliers and len(ranked[0]) > 2 else 0.0)
    if s > 0 and (b / s) < T_peak_ratio:
        return True, -1, b, (ranked[0][2] if use_inliers and len(ranked[0]) > 2 else 0.0)
    if z < T_z:
        return True, -1, b, (ranked[0][2] if use_inliers and len(ranked[0]) > 2 else 0.0)

    return False, ranked[0][0], b, (ranked[0][2] if use_inliers and len(ranked[0]) > 2 else 0.0)

    
def infer_num_paintings(
    ranked, 
    min_inliers: int = 15,      # each valid peak must exceed this
    min_ratio: float = 0.30,    # each valid peak must exceed this inlier ratio
    second_rel_to_first: float = 0.60,  # b2 >= 0.60 * b1  (not too weak)
    second_over_third: float = 1.4,     # b2/b3 dominance
    second_margin_abs: int = 6          # OR (b2 - b3) >= 6
) -> int:
    """
    ranked: [(img_id, inliers, inlier_ratio), ...] sorted desc by inliers.
    Returns 0, 1 or 2.
    """
    if not ranked:
        return 0

    # Unpack top-3 safely
    def _get(i, j):  # j: 0=id, 1=inliers, 2=ratio
        if len(ranked) > i and len(ranked[i]) > j:
            return ranked[i][j]
        return 0 if j == 1 else 0.0

    b1, r1 = _get(0, 1), _get(0, 2) # best match inliers and inlier ratio
    b2, r2 = _get(1, 1), _get(1, 2)
    b3, _  = _get(2, 1), _get(2, 2)

    # If best peak is weak → 0 paintings
    if b1 < min_inliers or r1 < min_ratio:
        return 0

    # Check if a second strong peak exists and is clearly distinct from the third
    has_second = (b2 >= min_inliers) and (r2 >= min_ratio) and (b2 / max(b1, 1e-6) >= second_rel_to_first)
    second_dominates_third = (b2 / max(b3, 1e-6) >= second_over_third) or ((b2 - b3) >= second_margin_abs)

    if has_second and second_dominates_third:
        return 2

    return 1

# Main method for finding top matches for a query
def find_top_ids_for_queries(
    queries_kp, 
    queries_desc,
    bbdd_kp, 
    bbdd_desc,
    paint_counts,
    desc="sift", 
    T_peak_ratio = 1.5,
    T_z = 3.0,
    k_stat = 50,
    backend="flann",
    splits=False,
    use_mutual=True, # bidirectional_ratio
    use_inliers=True, # Use RANSAC with homography       
    model="homography", 
    ransac_reproj=3.0,
    infer_from_inliers=True, # if True and use_inliers=True → infer 0/1/2 paintings
    T_inl=15,
    T_ratio=0.30,
    top_n=2
):
    """
    Returns: [[-1], [150], [48, 251], ...]
      - splits=True  → decide per crop (unknown or best id), then group per original using paint_counts
      - splits=False → infer 0/1/2 using inlier peaks; else fall back to calibrated 1-ID (or top_n) mode
    """
    results = []

    # State for grouping when splits=True
    pc_idx = 0
    pending_group = []
    # initialize remaining items in current group safely
    remaining_in_group = (paint_counts[0] if (splits and paint_counts and len(paint_counts) > 0) else 1)

    for kp_q, des_q in zip(queries_kp, queries_desc):
        if des_q is None or len(des_q) == 0:
            if splits:
                # only update the group; DO NOT append to results here
                pending_group.append(-1)
                remaining_in_group -= 1
                if remaining_in_group <= 0:
                    results.append(pending_group[:])
                    pending_group.clear()
                    pc_idx += 1
                    if paint_counts and pc_idx < len(paint_counts):
                        remaining_in_group = paint_counts[pc_idx]
                    else:
                        remaining_in_group = 1
            else:
                results.append([-1])
            continue

        ranked = rank_gallery(
            kp_query=kp_q, des_query=des_q,
            bbdd_kp=bbdd_kp, bbdd_desc=bbdd_desc,
            desc=desc, backend=backend,
            use_mutual=use_mutual,
            use_inliers=use_inliers,
            model=model, ransac_reproj=ransac_reproj
        )

        if splits:
            # Use the calibrated gate for a single best id (or unknown)
            is_unk, best_id, _, _ = decide_unknown_calibrated(
                ranked,
                use_inliers=use_inliers,
                T_inl_abs=T_inl,      
                T_ratio_abs=T_ratio,
                T_peak_ratio=T_peak_ratio,
                T_z=T_z,
                k_stat=k_stat
            )
            best_id = -1 if is_unk else best_id

            # Safe guard for paint_counts access
            current_group_size = paint_counts[pc_idx] if (paint_counts and pc_idx < len(paint_counts)) else 1

            if remaining_in_group == 1 and len(pending_group) == 0 and current_group_size == 1:
                # case 1: original image was not split → single id list
                results.append([best_id])
                pc_idx += 1
                if splits and paint_counts and pc_idx < len(paint_counts):
                    remaining_in_group = paint_counts[pc_idx]
                else:
                    remaining_in_group = 1
                continue
            else:
                # case 2: original image was split (or we are accumulating)
                pending_group.append(best_id)
                remaining_in_group -= 1
                if remaining_in_group == 0:
                    results.append(pending_group[:])  # e.g., [id_left, id_right]
                    pending_group.clear()
                    pc_idx += 1
                    if splits and paint_counts and pc_idx < len(paint_counts):
                        remaining_in_group = paint_counts[pc_idx]
                    else:
                        remaining_in_group = 1
                continue

        # splits == False 
        if use_inliers and infer_from_inliers:
            # infer 0/1/2 from inlier peaks
            n = infer_num_paintings(
                ranked,
                min_inliers=T_inl,
                min_ratio=T_ratio,
                second_rel_to_first=0.60,
                second_over_third=1.4,
                second_margin_abs=6
            )
            if n == 0:
                results.append([-1])
            elif n == 1:
                results.append([ranked[0][0]])
            else:  # n == 2
                ids = [ranked[0][0]]
                if len(ranked) > 1:
                    ids.append(ranked[1][0])
                results.append(ids)
        else:
            # Calibrated unknown vs accept; if accepted, return top_n 
            is_unk, best_id, _, _ = decide_unknown_calibrated(
                ranked,
                use_inliers=use_inliers,
                T_inl_abs=T_inl, T_ratio_abs=T_ratio,
                T_peak_ratio=T_peak_ratio, T_z=T_z, k_stat=k_stat
            )
            if is_unk:
                results.append([-1])
            else:
                results.append([idx for idx, *_ in ranked[:top_n]])

    return results


if __name__=="__main__":
    print("Read images")
    queries = read_images(SCRIPT_DIR.parent / "qsd1_w4")[:3]
    bbdd    = read_images(SCRIPT_DIR.parents[1] / "BBDD")

    print("Compute descriptors")
    keys_q, desc_q = compute_descriptors(queries, method="sift")
    keys_t, desc_t = compute_descriptors(bbdd,   method="sift")

    # Robust mode (RANSAC):
    # This is really slow, because we use sift + homography + bidirectional (mutual)!
    results = find_top_ids_for_queries(
        keys_q, desc_q, keys_t, desc_t)

    print("\nFinal results:")
    print(results)