# grid_search.py
from pathlib import Path
from typing import List, Dict, Tuple
import time

from utils.io_utils import read_images, read_pickle, write_pickle
from descriptors import compute_descriptors, deserialize_keypoints_list, serialize_keypoints_list
from main import process_images, to_py_int_results
from scoring import find_top_ids_for_queries
from gs_configs import (
    SIFT_CALIB_DEFAULTS,
    SIFT_INLIERS_GRID,
    ORB_INLIERS_GRID,
    ORB_CALIB_DEFAULTS,
    CONFIGS,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def evaluate_strict(predicted: List[List[int]], ground_truth: List[List[int]]) -> Dict[str, float]:
    assert len(predicted) == len(ground_truth)
    n = len(ground_truth)
    exact = sum(1 for p, g in zip(predicted, ground_truth) if list(p) == list(g))
    acc = exact / n if n else 0.0
    return {"accuracy": acc, "mean_precision": acc, "mAP@2": acc}

def diff_examples(predicted, ground_truth, limit=10):
    diffs = []
    for i, (p, g) in enumerate(zip(predicted, ground_truth)):
        if list(p) != list(g):
            diffs.append((i, g, p))
            if len(diffs) >= limit:
                break
    return diffs

def _desc_dir() -> Path:
    d = SCRIPT_DIR / "descriptors"
    d.mkdir(exist_ok=True)
    return d

def _kp_dir() -> Path:
    d = SCRIPT_DIR / "keypoints"
    d.mkdir(exist_ok=True)
    return d

def _paths_for(method: str) -> Tuple[Path, Path]:
    return (_desc_dir() / f"descriptors_{method}.pkl",
            _kp_dir() / f"keypoints_{method}.pkl")

def ensure_bbdd_features(method: str, bbdd_dir: Path) -> Tuple[list, list]:
    desc_path, keys_path = _paths_for(method)
    try:
        desc_bbdd = read_pickle(desc_path)
        keys_bbdd_serial = read_pickle(keys_path)
        keys_bbdd = deserialize_keypoints_list(keys_bbdd_serial)
        if len(desc_bbdd) != len(keys_bbdd):
            raise EOFError(f"Mismatch: desc={len(desc_bbdd)} keys={len(keys_bbdd)}")
        return keys_bbdd, desc_bbdd
    except FileNotFoundError:
        bbdd_images = read_images(bbdd_dir)
        keys_bbdd, desc_bbdd = compute_descriptors(bbdd_images, method=method, save_pkl=False)
        keys_serial = serialize_keypoints_list(keys_bbdd)
        write_pickle(desc_bbdd, desc_path)
        write_pickle(keys_serial, keys_path)
        return keys_bbdd, desc_bbdd

def backend_for_method(method: str) -> str:
    return "flann" if method in ("sift", "hsift") else "bf"


# Main grid search

def run_grid_search(
    q_dir: Path,
    bbdd_dir: Path,
    gt_path: Path,
    log_path: Path,
    limit_queries: int = None,
):
    # Load GT
    ground_truth = read_pickle(gt_path)

    # Prepare query images
    q_images = read_images(q_dir)
    if limit_queries is not None:
        q_images = q_images[:limit_queries]
        ground_truth = ground_truth[:limit_queries]

    # Log header
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Grid Search Results\n")

    # Cache per (method, splits) to avoid recomputing descriptors
    # value: (keys_q, desc_q, paint_counts, q_proc_images)
    query_cache: Dict[Tuple[str, bool], Tuple[list, list, list, list]] = {}

    for cfg in CONFIGS:
        name   = cfg["name"]
        method = cfg["method"]
        backend= cfg.get("backend", backend_for_method(method))

        # Ensure BBDD features
        keys_bbdd, desc_bbdd = ensure_bbdd_features(method, bbdd_dir=bbdd_dir)

        # Determine splits flag for this run (default False)
        # SIFT grid overrides may include 'splits'
        splits_flag = False
        # For SIFT_inliers branch we read from the override; for others, allow cfg to carry it
        if name != "SIFT_inliers":
            splits_flag = bool(cfg.get("splits", False))

        cache_key = (method, splits_flag)

        # Compute query features once per (method, splits)
        if cache_key not in query_cache:
            print(f"[{name}] Computing query descriptors for method={method}, splits={splits_flag} ...")
            # process_images from main: give it split flag; no BG for GS
            q_proc, paint_counts = process_images(q_images, split=splits_flag, background=False)
            # compute_descriptors should return (keys, desc, paint_counts) when splits=True
            keys_q, desc_q = compute_descriptors(q_proc, method=method, save_pkl=False)
            query_cache[cache_key] = (keys_q, desc_q, paint_counts, q_proc)
        else:
            keys_q, desc_q, paint_counts, q_proc = query_cache[cache_key]

        # SIFT with inliers + calibrated
        if name == "SIFT_inliers":
            for ov in SIFT_INLIERS_GRID:
                # Merge defaults + overrides
                p = {**SIFT_CALIB_DEFAULTS, **ov}
                splits_flag = bool(p.get("splits", SIFT_CALIB_DEFAULTS.get("splits", False)))
                # If this override changes splits, recompute/call from cache
                cache_key = (method, splits_flag)
                if cache_key not in query_cache:
                    print(f"[{name}|{ov.get('tag','')}] Recompute descriptors: splits={splits_flag}")
                    q_proc, _ = process_images(q_images, split=splits_flag, background=False)
                    keys_q, desc_q = compute_descriptors(q_proc, method=method, save_pkl=False)
                    query_cache[cache_key] = (keys_q, desc_q, paint_counts, q_proc)
                else:
                    keys_q, desc_q, paint_counts, q_proc = query_cache[cache_key]

                params = dict(
                    desc=method,
                    backend=backend,
                    use_mutual=p["use_mutual"],
                    use_inliers=p["use_inliers"],
                    model=p["model"],
                    ransac_reproj=p["ransac_reproj"],
                    # calibrated / inference
                    T_inl=p["T_inl"],
                    T_ratio=p["T_ratio"],
                    infer_from_inliers=p["infer_from_inliers"],
                    T_peak_ratio=p["T_peak_ratio"],
                    T_z=p["T_z"],
                    k_stat=p["k_stat"],
                    splits=splits_flag,
                )

                run_tag = (
                    f"{name}|{ov.get('tag','')}"
                    f"|splits={splits_flag}|Tinl={p['T_inl']}|Tr={p['T_ratio']:.2f}"
                    f"|rproj={p['ransac_reproj']:.1f}|mut={p['use_mutual']}"
                    f"|peak={p['T_peak_ratio']:.2f}|z={p['T_z']:.1f}|k={p['k_stat']}"
                )

                print(run_tag)
                t0 = time.time()
                results = find_top_ids_for_queries(keys_q, desc_q, keys_bbdd, desc_bbdd,
                                                   paint_counts=paint_counts, **params)
                dt = time.time() - t0
                results = to_py_int_results(results)

                metrics = evaluate_strict(results, ground_truth)
                mism = diff_examples(results, ground_truth, limit=10) if metrics["accuracy"] < 1.0 else []

                line = (
                    f"{run_tag} || acc={metrics['accuracy']:.3f}  "
                    f"prec={metrics['mean_precision']:.3f}  "
                    f"mAP@2={metrics['mAP@2']:.3f}  time={dt:.2f}s"
                )
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    if mism:
                        f.write("MISMATCH EXAMPLES (idx, gt, pred): " + repr(mism) + "\n")

        elif name == "ORB_inliers":
            for ov in ORB_INLIERS_GRID:
                p = {**ORB_CALIB_DEFAULTS, **ov}
                splits_flag = bool(p.get("splits", ORB_CALIB_DEFAULTS.get("splits", False)))

                cache_key = (method, splits_flag)
                if cache_key not in query_cache:
                    print(f"[{name}|{ov.get('tag','')}] Recompute descriptors: splits={splits_flag}")
                    q_proc, paint_counts = process_images(q_images, split=splits_flag, background=False)
                    # compute on already-split list; DO NOT split again here
                    keys_q, desc_q = compute_descriptors(q_proc, method=method, save_pkl=False)
                    query_cache[cache_key] = (keys_q, desc_q, paint_counts, q_proc)
                else:
                    keys_q, desc_q, paint_counts, q_proc = query_cache[cache_key]
                params = dict(
                    desc=method,
                    backend=backend,
                    use_mutual=p["use_mutual"],
                    use_inliers=p["use_inliers"],
                    model=p["model"],
                    ransac_reproj=p["ransac_reproj"],
                    T_inl=p["T_inl"],
                    T_ratio=p["T_ratio"],
                    T_peak_ratio=p["T_peak_ratio"],
                    T_z=p["T_z"],
                    k_stat=p["k_stat"],
                    top_n=p["top_n"],
                    infer_from_inliers=p["infer_from_inliers"],
                    splits=splits_flag,
                )

                run_tag = (
                    f"{name}|{ov.get('tag','')}"
                    f"|splits={splits_flag}|Tinl={p['T_inl']}|Tr={p['T_ratio']:.2f}"
                    f"|rproj={p['ransac_reproj']:.1f}|mut={p['use_mutual']}"
                    f"|peak={p['T_peak_ratio']:.2f}|z={p['T_z']:.1f}|k={p['k_stat']}"
                )

                print(run_tag)
                t0 = time.time()
                results = find_top_ids_for_queries(keys_q, desc_q, keys_bbdd, desc_bbdd,
                                                   paint_counts=paint_counts, **params)
                dt = time.time() - t0
                results = to_py_int_results(results)

                metrics = evaluate_strict(results, ground_truth)
                mism = diff_examples(results, ground_truth, limit=10) if metrics["accuracy"] < 1.0 else []

                line = (
                    f"{run_tag} || acc={metrics['accuracy']:.3f}  "
                    f"prec={metrics['mean_precision']:.3f}  "
                    f"mAP@2={metrics['mAP@2']:.3f}  time={dt:.2f}s"
                )
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    if mism:
                        f.write("MISMATCH EXAMPLES (idx, gt, pred): " + repr(mism) + "\n")

        # Other configs (baseline path) 
        else:
            splits_flag = bool(cfg.get("splits", False))
            cache_key = (method, splits_flag)
            if cache_key not in query_cache:
                print(f"[{name}] Recompute descriptors: splits={splits_flag}")
                q_proc, _ = process_images(q_images, split=splits_flag, background=False)
                keys_q, desc_q = compute_descriptors(q_proc, method=method, save_pkl=False)
                query_cache[cache_key] = (keys_q, desc_q, paint_counts, q_proc)
            else:
                keys_q, desc_q, paint_counts, q_proc = query_cache[cache_key]

            params = dict(
                desc=method,
                backend=backend,
                use_mutual=cfg.get("use_mutual", True),
                use_inliers=cfg.get("use_inliers", False),
                model=cfg.get("model", "homography" if method != "orb" else "affine"),
                ransac_reproj=cfg.get("ransac_reproj", 3.0),
                T_inl=cfg.get("T_inl", 15),
                T_ratio=cfg.get("T_ratio", 0.30),
                top_n=2,
                infer_from_inliers=True,
                splits=splits_flag,
            )

            run_tag = f"{name}|method={method}|splits={splits_flag}"
            print(run_tag)
            t0 = time.time()
            results = find_top_ids_for_queries(keys_q, desc_q, keys_bbdd, desc_bbdd,
                                               paint_counts=paint_counts, **params)
            dt = time.time() - t0
            results = to_py_int_results(results)

            metrics = evaluate_strict(results, ground_truth)
            mism = diff_examples(results, ground_truth, limit=10) if metrics["accuracy"] < 1.0 else []

            line = (
                f"{run_tag} || acc={metrics['accuracy']:.3f}  "
                f"prec={metrics['mean_precision']:.3f}  "
                f"mAP@2={metrics['mAP@2']:.3f}  time={dt:.2f}s"
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                if mism:
                    f.write("MISMATCH EXAMPLES (idx, gt, pred): " + repr(mism) + "\n")


if __name__ == "__main__":
    Q_DIR = SCRIPT_DIR.parent / "qsd1_w4"
    BBDD_DIR = SCRIPT_DIR.parent.parent / "BBDD"
    GT_PATH = Q_DIR / "gt_corresps.pkl"
    LOGPATH = SCRIPT_DIR / "outputs_test" / "grid_search_log.txt"

    run_grid_search(
        q_dir=Q_DIR,
        bbdd_dir=BBDD_DIR,
        gt_path=GT_PATH,
        log_path=LOGPATH,
        limit_queries=None
    )
