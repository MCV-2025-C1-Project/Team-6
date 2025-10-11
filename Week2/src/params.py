"""This scripts contains parameters for Week's 2 experiments."""

# Grid Seach for best descriptors
descriptor_experiments = {
    "k_values": [1,5],
    "methods": ["hsv"], 
    "n_bins": [16],
    "metrics": [ "l1", "chi2", ],
    "n_crops": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
    "pyramid_levels": [[1,3,5,7,10,13,15,17,20,23],[3,5,7,10,13,15,17,20,23],[5,7,10,13,15,17,20,23],[7,10,13,15,17,20,23],[13,15,17,20,23],[15,17,20,23],[17,20,23],[20,23],[23],],
}

best_config_descriptors = {
    "k_values": 1,
    "color_space": "hsv", 
    "n_bins": 16,
    "metric": "l1",
    "n_crops": 23,
    "pyramid": False
}

best_config1 = {
    "method": "hsv",
    "n_bins": 16,
    "metric": "chi2"
}
best_config2 = {
    "method": "rgb-hs", 
    "n_bins": 16,
    "metric": "l1"
}

# Grid Search for best segmentation technique
segmentation_experiments = {
    "color_spaces": ["lab", "hsv"],
    "border_widths": [5, 10, 20], 
    "use_percentile_thresh": [True, False],
    "percentiles": [97, 99],
    "cov_fractions": [0.75, 0.9],
    "angle_limits": [0, 15, 30],
    "lambda_penalties": [1.0, 2.0],
    "min_fracs": [0.5, 0.8],
    "steps": [4],
    "use_best_square": [True, False]
}

best_config_segmentation = {
    'color_space': 'lab', 
    'border_width': 20, 
    'use_percentile_thresh': False, 
    'percentile': 99, 
    'cov_fraction': 0.9,  
    'angle_limit':  0, #15
    'lambda_penalty': 2.0, 
    'min_frac': 0.5, 'step': 4, 
    'use_best_square': True
    }

