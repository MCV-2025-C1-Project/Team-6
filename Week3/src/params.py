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

# Parameters for DCT Grid Search
dct_search_space = {
    "method": ["dct-rgbg"],
    "n_crops": [1,2],
    "n_coefs": [150,160,165,170,175,180]
}

# Best parameters for descriptor DCT
best_desc_params_dct = {
    "method": "dct-rgb",
    "n_crops": 1,
    "n_coefs": 150
}

# NOTHING TO ADD BECAUSE OF ITS IRRELEVANCE
# Parameters for LBP Grid Search
lbp_search_space = {

}

# Best parameters for descriptor LBP
best_desc_params_lbp = {

}


