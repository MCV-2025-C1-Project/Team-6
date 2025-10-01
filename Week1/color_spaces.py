import numpy as np
"""Implementations of color space conversions based on the website: http://brucelindbloom.com/ 
and the website: https://www.rapidtables.com/convert/color/rgb-to-hsv.html"""

def rgb_to_xyz(rgb):
    # We assume sRGB, as its not specified
    M = np.array([[0.4124564  , 0.3575761  , 0.1804375],
                  [0.2126729  , 0.7151522  ,0.0721750 ],
                  [0.0193339   , 0.1191920   ,0.9503041  ]])

    XYZ = np.dot(M, rgb)
    return XYZ

def rgb_to_hsv(rgb):
    rgb = rgb / 255.0
    Cmax = np.max(rgb)
    Cmin = np.min(rgb)
    delta = Cmax - Cmin

    # Hue calculation
    if delta == 0:
        H = 0
    elif Cmax == rgb[0]:
        H = 60 * (((rgb[1] - rgb[2]) / delta) % 6)
    elif Cmax == rgb[1]:
        H = 60 * (((rgb[2] - rgb[0]) / delta) + 2)
    elif Cmax == rgb[2]:
        H = 60 * (((rgb[0] - rgb[1]) / delta) + 4)

    # Saturation calculation
    if Cmax == 0:
        S = 0
    else:
        S = delta / Cmax

    # Value calculation
    V = Cmax
    
    return np.array([H, S * 100, V * 100])