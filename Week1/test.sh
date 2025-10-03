#!/bin/bash

# NOT TO MUCH MODULARIZED

python3 Week1/main.py -k 5 -pkl ./BBDD/BBDD_descriptors_rgb.pkl -dist euclidean -plt query_rgb_plot.png -desc rgb

python3 Week1/main.py -k 5 -pkl ./BBDD/BBDD_descriptors_hsv.pkl -dist euclidean -plt query_hsv_plot.png -desc hsv

python3 Week1/main.py -k 5 -pkl ./BBDD/BBDD_descriptors_hs.pkl -dist euclidean -plt query_hs_plot.png -desc hs

python3 Week1/main.py -k 5 -pkl ./BBDD/BBDD_descriptors_hs_rgb.pkl -dist euclidean -plt query_hs_rgb_plot.png -desc hs_rgb