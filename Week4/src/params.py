FLANN_MATCHER_PARAMS = {
    "index_params": {
        "sift": dict(algorithm=1, trees=5),
        "hsift": dict(algorithm=1, trees=5),
        "orb": dict(algorithm= 6,table_number= 6, key_size= 12, multi_probe_level= 1)
    },
    "SearchParams": dict(checks=50)
}

BEST_DESCRIPTOR_PARAMS = {
    "method": "sift",         
    "backend": "flann", #bf for orb
    "use_mutual": True, #bidirectional ratio for cleaner matches
    "use_inliers": True, # rank by RANSAC inliers 
    # (RANSAC) 
    "model": "homography",       
    "ransac_reproj": 3.0,
    # Calibrated unknown logic ---
    "T_inl": 15,                
    "T_ratio": 0.35,             
    "T_peak_ratio": 1.6,        
    "T_z": 3.2,                  
    "k_stat": 50,              
    # Preprocessing flags for main
    "split": True,              
    "background": False
}


# Noise removal parameters

# Base thresholds for each noise type for the grid search
base_thresholds = {
    "sp_impulse": 0.06,
    "sig_gauss":  0.05,
    "blk_jpeg":   1.30,
    "chr_chroma": 0.25,
    "vol_blur":   80.0
}

# Best noise parameters to test the queries set
BEST_NOISE_PARAMS = {
    "sp_impulse": 0.08,   
    "sig_gauss":  0.05,   
    "blk_jpeg":   1.15,   
    "chr_chroma": 0.25,   
    "vol_blur":   80.0    
}

# Parameteres for Noise Grid Search
noise_search_space = {
    "sp_impulse": [0.04, 0.08, 0.1],
    "sig_gauss":  [0.05, 0.06, 0.07],
    "blk_jpeg":   [1.15, 1.30],
    "chr_chroma": [0.25, 0.30],
    "vol_blur": [80.0]
}   