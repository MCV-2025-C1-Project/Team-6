# grid_search.py
from pathlib import Path
from typing import List, Dict, Tuple
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import traceback
from typing import Dict, Tuple, List, Any
from pathlib import Path
import time
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import traceback
from typing import Dict, Tuple, List, Any
from pathlib import Path
import time
import threading



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

# Thread-safe lock for file writing
log_lock = threading.Lock()


def process_single_config(
    cfg: dict,
    ground_truth: list,
    bbdd_dir: Path,
    query_cache: dict,
) -> dict:
    """
    Process a single configuration (non-inliers baseline).
    Returns result dict with metrics or error info.
    """
    try:
        name = cfg["name"]
        method = cfg["method"]
        backend = cfg.get("backend", backend_for_method(method))
        
        # Get cached query features
        splits_flag = bool(cfg.get("splits", False))
        cache_key = (method, splits_flag)
        
        if cache_key not in query_cache:
            raise ValueError(f"Cache key {cache_key} not found for config {name}")
        
        keys_q, desc_q, paint_counts, q_proc = query_cache[cache_key]
        
        # Ensure BBDD features
        keys_bbdd, desc_bbdd = ensure_bbdd_features(method, bbdd_dir=bbdd_dir)
        
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
        
        t0 = time.time()
        results = find_top_ids_for_queries(
            keys_q, desc_q, keys_bbdd, desc_bbdd,
            paint_counts=paint_counts, **params
        )
        dt = time.time() - t0
        results = to_py_int_results(results)
        
        metrics = evaluate_strict(results, ground_truth)
        mism = diff_examples(results, ground_truth, limit=10) if metrics["accuracy"] < 1.0 else []
        
        return {
            "success": True,
            "run_tag": run_tag,
            "metrics": metrics,
            "mismatches": mism,
            "time": dt,
        }
        
    except Exception as e:
        return {
            "success": False,
            "run_tag": cfg.get("name", "unknown"),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def process_single_inliers_config(
    cfg_name: str,
    method: str,
    backend: str,
    override: dict,
    defaults: dict,
    ground_truth: list,
    bbdd_dir: Path,
    query_cache: dict,
) -> dict:
    """
    Process a single SIFT/ORB inliers configuration.
    Returns result dict with metrics or error info.
    """
    try:
        # Merge defaults + overrides
        p = {**defaults, **override}
        splits_flag = bool(p.get("splits", defaults.get("splits", False)))
        
        # Get cached query features
        cache_key = (method, splits_flag)
        if cache_key not in query_cache:
            raise ValueError(f"Cache key {cache_key} not found for {cfg_name}")
        
        keys_q, desc_q, paint_counts, q_proc = query_cache[cache_key]
        
        # Ensure BBDD features
        keys_bbdd, desc_bbdd = ensure_bbdd_features(method, bbdd_dir=bbdd_dir)
        
        params = dict(
            desc=method,
            backend=backend,
            use_mutual=p["use_mutual"],
            use_inliers=p["use_inliers"],
            model=p["model"],
            ransac_reproj=p["ransac_reproj"],
            T_inl=p["T_inl"],
            T_ratio=p["T_ratio"],
            infer_from_inliers=p["infer_from_inliers"],
            T_peak_ratio=p["T_peak_ratio"],
            T_z=p["T_z"],
            k_stat=p["k_stat"],
            splits=splits_flag,
        )
        
        # Add top_n for ORB
        if cfg_name == "ORB_inliers":
            params["top_n"] = p["top_n"]
        
        run_tag = (
            f"{cfg_name}|{override.get('tag','')}"
            f"|splits={splits_flag}|Tinl={p['T_inl']}|Tr={p['T_ratio']:.2f}"
            f"|rproj={p['ransac_reproj']:.1f}|mut={p['use_mutual']}"
            f"|peak={p['T_peak_ratio']:.2f}|z={p['T_z']:.1f}|k={p['k_stat']}"
        )
        
        t0 = time.time()
        results = find_top_ids_for_queries(
            keys_q, desc_q, keys_bbdd, desc_bbdd,
            paint_counts=paint_counts, **params
        )
        dt = time.time() - t0
        results = to_py_int_results(results)
        
        metrics = evaluate_strict(results, ground_truth)
        mism = diff_examples(results, ground_truth, limit=10) if metrics["accuracy"] < 1.0 else []
        
        return {
            "success": True,
            "run_tag": run_tag,
            "metrics": metrics,
            "mismatches": mism,
            "time": dt,
        }
        
    except Exception as e:
        return {
            "success": False,
            "run_tag": f"{cfg_name}|{override.get('tag', 'unknown')}",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def compute_query_cache(
    q_images: list,
    configs: list,
    sift_inliers_grid: list,
    orb_inliers_grid: list,
    sift_defaults: dict,
    orb_defaults: dict,
) -> Dict[Tuple[str, bool], Tuple[list, list, list, list]]:
    """
    Pre-compute all query descriptors for all (method, splits) combinations.
    """
    query_cache = {}
    cache_keys_needed = set()
    
    # Collect all cache keys needed
    for cfg in configs:
        method = cfg["method"]
        name = cfg["name"]
        
        if name == "SIFT_inliers":
            for ov in sift_inliers_grid:
                p = {**sift_defaults, **ov}
                splits_flag = bool(p.get("splits", sift_defaults.get("splits", False)))
                cache_keys_needed.add((method, splits_flag))
        elif name == "ORB_inliers":
            for ov in orb_inliers_grid:
                p = {**orb_defaults, **ov}
                splits_flag = bool(p.get("splits", orb_defaults.get("splits", False)))
                cache_keys_needed.add((method, splits_flag))
        else:
            splits_flag = bool(cfg.get("splits", False))
            cache_keys_needed.add((method, splits_flag))
    
    # Compute all needed cache entries
    for method, splits_flag in cache_keys_needed:
        try:
            print(f"Computing query descriptors for method={method}, splits={splits_flag} ...")
            q_proc, paint_counts = process_images(q_images, split=splits_flag, background=False)
            keys_q, desc_q = compute_descriptors(q_proc, method=method, save_pkl=False)
            query_cache[(method, splits_flag)] = (keys_q, desc_q, paint_counts, q_proc)
        except Exception as e:
            print(f"ERROR computing cache for {method}, splits={splits_flag}: {e}")
            traceback.print_exc()
    
    return query_cache


def write_result_to_log(log_path: Path, result: dict):
    """Write a single result to the log file (thread-safe)."""
    try:
        with log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                if result["success"]:
                    line = (
                        f"{result['run_tag']} || acc={result['metrics']['accuracy']:.3f}  "
                        f"prec={result['metrics']['mean_precision']:.3f}  "
                        f"mAP@2={result['metrics']['mAP@2']:.3f}  time={result['time']:.2f}s"
                    )
                    f.write(line + "\n")
                    if result["mismatches"]:
                        f.write("MISMATCH EXAMPLES (idx, gt, pred): " + repr(result["mismatches"]) + "\n")
                else:
                    f.write(f"ERROR in {result['run_tag']}: {result['error']}\n")
                    f.write(f"Traceback:\n{result['traceback']}\n")
    except Exception as e:
        print(f"Failed to write to log: {e}")


def run_grid_search(
    q_dir: Path,
    bbdd_dir: Path,
    gt_path: Path,
    log_path: Path,
    limit_queries: int = None,
    max_workers: int = None,
):
    """
    Parallelized grid search with error handling using threads.
    
    Args:
        max_workers: Maximum number of parallel workers (None = CPU count * 5)
    """
    try:
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
            f.write("Grid Search Results (Parallelized)\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
        
        # Pre-compute all query descriptors (sequential, as it's I/O bound)
        print("Pre-computing query descriptors for all configurations...")
        query_cache = compute_query_cache(
            q_images, CONFIGS, 
            SIFT_INLIERS_GRID, ORB_INLIERS_GRID,
            SIFT_CALIB_DEFAULTS, ORB_CALIB_DEFAULTS
        )
        
        print(f"\nStarting parallelized grid search with {max_workers or 'auto'} workers...")
        
        # Collect all tasks
        tasks = []
        
        for cfg in CONFIGS:
            name = cfg["name"]
            method = cfg["method"]
            backend = cfg.get("backend", backend_for_method(method))
            
            if name == "SIFT_inliers":
                # Add all SIFT inliers configurations
                for ov in SIFT_INLIERS_GRID:
                    tasks.append({
                        "type": "inliers",
                        "cfg_name": name,
                        "method": method,
                        "backend": backend,
                        "override": ov,
                        "defaults": SIFT_CALIB_DEFAULTS,
                    })
                    
            elif name == "ORB_inliers":
                # Add all ORB inliers configurations
                for ov in ORB_INLIERS_GRID:
                    tasks.append({
                        "type": "inliers",
                        "cfg_name": name,
                        "method": method,
                        "backend": backend,
                        "override": ov,
                        "defaults": ORB_CALIB_DEFAULTS,
                    })
            else:
                # Add baseline configuration
                tasks.append({
                    "type": "baseline",
                    "cfg": cfg,
                })
        
        print(f"Total configurations to run: {len(tasks)}")
        
        # Process tasks in parallel using threads
        completed = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            
            for task in tasks:
                if task["type"] == "inliers":
                    future = executor.submit(
                        process_single_inliers_config,
                        task["cfg_name"],
                        task["method"],
                        task["backend"],
                        task["override"],
                        task["defaults"],
                        ground_truth,
                        bbdd_dir,
                        query_cache,
                    )
                else:  # baseline
                    future = executor.submit(
                        process_single_config,
                        task["cfg"],
                        ground_truth,
                        bbdd_dir,
                        query_cache,
                    )
                
                future_to_task[future] = task
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    completed += 1
                    
                    if result["success"]:
                        print(f"[{completed}/{len(tasks)}] ✓ {result['run_tag']}")
                    else:
                        failed += 1
                        print(f"[{completed}/{len(tasks)}] ✗ {result['run_tag']} - ERROR")
                    
                    # Write to log immediately
                    write_result_to_log(log_path, result)
                    
                except Exception as e:
                    failed += 1
                    completed += 1
                    task = future_to_task[future]
                    error_result = {
                        "success": False,
                        "run_tag": str(task),
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    print(f"[{completed}/{len(tasks)}] ✗ {task} - EXCEPTION")
                    write_result_to_log(log_path, error_result)
        
        # Write summary
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write(f"Completed: {completed}/{len(tasks)}\n")
            f.write(f"Failed: {failed}/{len(tasks)}\n")
            f.write(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"\nGrid search complete! Completed: {completed}/{len(tasks)}, Failed: {failed}")
        print(f"Results saved to: {log_path}")
        
    except Exception as e:
        print(f"FATAL ERROR in run_grid_search: {e}")
        traceback.print_exc()
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\nFATAL ERROR: {e}\n")
                f.write(traceback.format_exc() + "\n")
        except:
            pass
        raise


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
