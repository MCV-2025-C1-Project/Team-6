import numpy as np
import cv2 as cv
from pathlib import Path
from typing import Tuple

from descriptors import compute_descriptors
from utils.io_utils import read_images
from matching import bidirectional_ratio_matches, ratio_matches, bf, flann

SCRIPT_DIR = Path(__file__).resolve().parent


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
        matches = ratio_matches(des_query, des_target, desc=desc, backend=backend, ratio=ratio, matcher=matcher_fwd )
    if len(matches) < 8:
        return 0, 0.0
    
    # Cap matches so ransac is much faster
    matches = matches[:400]

    # geometric verification
    # cv.findHomography() expects the points in shape (N, 1, 2) (now they are Nx2)
    # src = np.float32([kp_query[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    # dst = np.float32([kp_target[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
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
    return inliers, inlier_ratio

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
            prelim = (bidirectional_ratio_matches if use_mutual else ratio_matches)(
                des_query, des_t, desc=desc, backend=backend, ratio=ratio, top_k=300,
                matcher_fwd=matcher_fwd if use_mutual else None,
                matcher_rev=matcher_rev if use_mutual else None,
                matcher=matcher_fwd if not use_mutual else None
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
                                  matcher_rev=matcher_rev )
            scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

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
    second = ranked[1][1] if len(ranked) > 1 else 0

    if use_inliers:
        inl_ratio = ranked[0][2] if len(ranked[0]) > 2 else 0
        is_unknown = (best < T_inl) or (inl_ratio < T_ratio) or ((best - second) < margin)
        return is_unknown, (-1 if is_unknown else best_id), best, inl_ratio
    else:
        is_unknown = (best < T_matches) or ((best - second) < margin)
        return is_unknown, (-1 if is_unknown else best_id), best, second
    
def infer_num_paintings(ranked, min_inliers=15, ratio_drop=0.6):
    """
    ranked: [(img_id, inliers, inlier_ratio), ...] sorted desc by inliers.
    Returns 0, 1 or 2.
    """
    if not ranked:
        return 0
    best = ranked[0][1] if len(ranked[0]) > 1 else 0
    second = ranked[1][1] if len(ranked) > 1 and len(ranked[1]) > 1 else 0
    if best < min_inliers:
        return 0
    if second >= min_inliers and (second / max(best, 1)) > ratio_drop:
        return 2
    return 1
    
def find_top_ids_for_queries(
    queries_kp, 
    queries_desc,
    bbdd_kp, 
    bbdd_desc,
    desc="sift", 
    backend="flann",
    use_mutual=True,
    use_inliers=True, # if False, we just return top_n ids (no inference)         
    model="homography", ransac_reproj=3.0,
    infer_from_inliers=True, #decide 0/1/2 paintings automatically
    T_matches=12, T_inl=15, T_ratio=0.30, margin=3,
    top_n=2,
    infer_ratio_drop=0.6):
    """
    Returns: [[-1], [150], [48, 251], ...]
    """
    results = []
    for kp_q, des_q in zip(queries_kp, queries_desc):
        if des_q is None or len(des_q) == 0:
            print("No descriptor was computed.")
            results.append([-1]); continue

        ranked = rank_gallery(
            kp_query=kp_q, des_query=des_q,
            bbdd_kp=bbdd_kp, bbdd_desc=bbdd_desc,
            desc=desc, backend=backend,
            use_mutual=use_mutual,
            use_inliers=use_inliers,
            model=model, ransac_reproj=ransac_reproj
        )

        if use_inliers and infer_from_inliers and top_n != 1:
            # Decide 0/1/2 purely from inlier peaks
            n = infer_num_paintings(ranked, min_inliers=T_inl, ratio_drop=infer_ratio_drop)
            if n == 0:
                results.append([-1])
            elif n == 1:
                results.append([ranked[0][0]])
            else:  # n == 2
                # guard if db has only one candidate
                ids = [ranked[0][0]]
                if len(ranked) > 1:
                    ids.append(ranked[1][0])
                results.append(ids)
        else:
            # Fallback to threshold decision logic
            is_unk, _, _, _ = decide_unknown(
                ranked,
                use_inliers=use_inliers,
                T_matches=T_matches, T_inl=T_inl, T_ratio=T_ratio, margin=margin
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
        keys_q, desc_q, keys_t, desc_t,
        desc="sift", 
        backend="flann",
        use_mutual=True,
        use_inliers=True,               
        model="homography", ransac_reproj=3.0,
        T_inl=15, T_ratio=0.30, margin=3,
        top_n=2
    )

    print("\nFinal results:")
    print(results)