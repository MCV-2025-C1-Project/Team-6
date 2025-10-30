FLANN_MATCHER_PARAMS = {
    "index_params": {
        "sift": dict(algorithm=1, tress=5),
        "orb": dict(algorithm= 6,table_number= 6, key_size= 12, multi_probe_level= 1)
    },
    "SearchParams": dict(checks=50)
}

BEST_DESCRIPTOR_PARAMS = {
    "method": "sift"
} #this has not been tested yet...

BEST_NOISE_PARAMS = {
    "sp_impulse": 0.08,   
    "sig_gauss":  0.05,   
    "blk_jpeg":   1.15,   
    "chr_chroma": 0.25,   
    "vol_blur":   80.0 
}