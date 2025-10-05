"""This scripts contains parameters for Week's 1 experiments."""
experiments = {
    "k_values": [1,5],
    "methods": ["rgb", "hs", "hsv", "rgb-hs", "rgb-hsv"], 
    "n_bins": [16, 32, 64, 128, 256],
    "metrics": ["euclidean", "l1", "chi2", "histogram_intersection", "hellinger", "cosine", "bhattacharyya"]
}
