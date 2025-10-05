"""This scripts contains parameters for Week's 1 experiments."""
experiments = {
    "k_value": 1, # 1 or 5
    "methods": ["rgb", "hs", "hsv", "rgb-hs", "rgb-hsv"], 
    "n_bins": [16, 32, 64, 128, 256],
    "metrics": ["euclidean", "l1", "chi2", "histogram_intersection", "hellinger", "cosine", "bhattacharyya"]
}

best_config_1 = {
    "k_value": 1,
    "methods": ["hs"],
    "n_bins": [128],
    "metrics": ["l1"]
}

best_config_5 = {
    "k_value": 5,
    "methods": ["hsv"],
    "n_bins": [16],
    "metrics": ["l1"]
}