from typing import Dict, List

CONFIGS = [
    # SIFT + FLANN + inliers + calibrated
    dict(name="SIFT_inliers", method="sift", backend="flann",
         use_inliers=True, use_mutual=True,
         model="homography", ransac_reproj=3.0,
         T_inl=15, T_ratio=0.35),

    # ORB + BF + inliers + calibrated
    dict(name="ORB_inliers", method="orb", backend="bf",
         use_inliers=True, use_mutual=True,
         model="affine", ransac_reproj=5.0,
         T_inl=10, T_ratio=0.30),
]

#SIFT calibrated defaults + grid
SIFT_CALIB_DEFAULTS: Dict = dict(
    use_mutual=True,
    use_inliers=True,
    model="homography",
    top_n=2,
    infer_from_inliers=True,
    ransac_reproj=3.0,
    T_inl=15,
    T_ratio=0.35,
    T_peak_ratio=1.6,
    T_z=3.2,
    k_stat=50,
    splits=False,
)

SIFT_INLIERS_GRID: List[Dict] = [
    dict(tag="E1_baseline"),
    dict(tag="E2_rproj4",     ransac_reproj=4.0),
    dict(tag="E3_Tinl12",     T_inl=12),
    dict(tag="E4_Tratio040",  T_ratio=0.40),
    dict(tag="E5_split",     splits=True),
]

# ORB calibrated defaults + grid
ORB_CALIB_DEFAULTS: Dict = dict(
    use_mutual=True,
    use_inliers=True,
    model="affine",
    ransac_reproj=5.0,
    T_inl=12,        # ORB tends to need a bit more to be confident
    T_ratio=0.30,
    T_peak_ratio=1.6,
    T_z=3.0,
    k_stat=50,
    top_n=2,
    infer_from_inliers=True,   # if splits=False, allow 0/1/2 inference
    splits=False,
)

ORB_INLIERS_GRID: List[Dict] = [
    dict(tag="O1_baseline_no_split", splits=False),
    dict(tag="O2_with_split",        splits=True),
    dict(tag="O3_rproj6", ransac_reproj=6.0),
    dict(tag="O4_Tinl14", T_inl=14),
]