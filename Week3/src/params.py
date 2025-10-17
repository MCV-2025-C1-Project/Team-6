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
best_noise_params = {
    "sp_impulse": 0.06,   
    "sig_gauss":  0.05,   
    "blk_jpeg":   1.30,   
    "chr_chroma": 0.25,   
    "vol_blur":   80.0    
}

# Parameteres for the Grid Search
noise_search_space = {
    "sp_impulse": [0.06, 0.08, 0.10],
    "sig_gauss":  [0.05, 0.06, 0.07],
    "blk_jpeg":   [1.30, 1.35, 1.40],
    "chr_chroma": [0.25, 0.30, 0.35],
    "vol_blur": [80.0]
}   


best_desc_params = {
    "method": "dct-rgb",
    "n_crops": 4,
    "n_coefs": 80
}


