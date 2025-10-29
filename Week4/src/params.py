FLANN_MATCHER_PARAMS = {
    "index_params": {
        "sift": dict(algorithm=1, tress=5),
        "orb": dict(algorithm= 6,table_number= 6, key_size= 12, multi_probe_level= 1)
    },
    "SearchParams": dict(checks=50)
}