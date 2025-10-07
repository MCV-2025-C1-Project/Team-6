"""This scripts contains parameters for Week's 1 experiments."""
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