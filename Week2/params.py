"""This scripts contains parameters for Week's 1 experiments."""

# TODO:  Do we need all of those?

# TODO:  Do we need all of those?
experiments = {
    "k_values": [1,5],
    "methods": ["hsv"], 
    "n_bins": [16],
    "metrics": [ "l1", "chi2", ],
    "n_crops": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
    "pyramid_levels": [[1,10,20], [1,5,10,15,20], [1,3,5,7,9,11,13,15,17,19],[1, 4, 7, 10, 13, 16, 19], [1, 6, 11, 16, 20], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
}
# 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
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

background_best_parameters = {'color_space': 'lab', 'border_width': 20, 'use_percentile_thresh': False, 'percentile': 99, 'cov_fraction': 0.9,  'angle_limit': 15, 'lambda_penalty': 2.0, 'min_frac': 0.5, 'step': 4, 'use_best_square': True}

import numpy as np
import itertools

def create_grid_search_experiments(
    color_spaces=["lab", "hsv"],
    border_widths=[5, 10, 20],
    use_percentile_thresh=[True, False],
    percentiles=[97, 99],
    cov_fractions=[0.75, 0.9],
    angle_limits=[0, 15, 30],
    lambda_penalties=[1.0, 2.0],
    min_fracs=[0.5, 0.8],
    steps=[4],
    use_best_square=[True, False]    
):
    experiments = []

    for (color_space, border_width, use_perc, percentile, cov_fraction,
         angle_limit, lambda_penalty, min_frac, step, use_bs) in itertools.product(
            color_spaces, border_widths, use_percentile_thresh, percentiles,
            cov_fractions, angle_limits, lambda_penalties, min_fracs, steps, use_best_square
    ):
        exp = {
            "color_space": color_space,
            "border_width": border_width,
            "use_percentile_thresh": use_perc,
            "percentile": percentile,
            "cov_fraction": cov_fraction,
            "angle_limit": angle_limit,
            "lambda_penalty": lambda_penalty,
            "min_frac": min_frac,
            "step": step,
            "use_best_square": use_bs
        }
        
        experiments.append(exp)
    
    return experiments

