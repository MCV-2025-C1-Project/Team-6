"""This scripts contains parameters for Week's 1 experiments."""

# TODO:  Do we need all of those?
experiments = {
    "k_values": [1,5],
    "methods": ["hsv"], 
    "n_bins": [16],
    "metrics": [ "l1", "chi2", ],
    "n_crops": [2,3,4,5,6,7,8],
}

# Best configurations
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

# Maybe it would be nicer to have a real grid search...
background_experiments = [
    {"color_space": "lab", "border_width": 10, "use_percentile_thresh": True,  "percentile": 97, "cov_fraction": 0.75, "angle_limit": 15, "angle_step": 1, "lambda_penalty": 1.0, "min_frac": 0.5, "step": 4},
    {"color_space": "lab", "border_width": 20, "use_percentile_thresh": True,  "percentile": 99, "cov_fraction": 0.9,"angle_limit": 30, "angle_step": 1, "lambda_penalty": 1.2, "min_frac": 0.5, "step": 4},
    {"color_space": "hsv", "border_width": 10, "use_percentile_thresh": True,  "percentile": 99, "cov_fraction": 0.75,"angle_limit": 30, "angle_step": 1, "lambda_penalty": 1.0, "min_frac": 0.5, "step": 4},
    {"color_space": "hsv", "border_width": 20, "use_percentile_thresh": True,  "percentile": 97, "cov_fraction": 0.9,"angle_limit": 15, "angle_step": 1, "lambda_penalty": 1.2, "min_frac": 0.5, "step": 4},
    {"color_space": "lab", "border_width": 10, "use_percentile_thresh": False,                   "cov_fraction": 0.75,"angle_limit": 15, "angle_step": 2, "lambda_penalty": 1.0, "min_frac": 0.8, "step": 4},
    {"color_space": "hsv", "border_width": 20, "use_percentile_thresh": False,                   "cov_fraction": 0.9,"angle_limit": 30, "angle_step": 2, "lambda_penalty": 1.2, "min_frac": 0.8, "step": 4},
]