import numpy as np
"""Implementations of color space conversions based on the website: http://brucelindbloom.com/ 
and the website: https://www.rapidtables.com/convert/color/rgb-to-hsv.html"""

def rgb_to_xyz(rgb):
    og_shape = rgb.shape
    rgb = rgb.reshape(3, -1) / 255.0
    # We assume sRGB, as its not specified
    M = np.array([[0.4124564  , 0.3575761  , 0.1804375],
                  [0.2126729  , 0.7151522  ,0.0721750 ],
                  [0.0193339   , 0.1191920   ,0.9503041  ]])

    XYZ = np.dot(M, rgb)
    XYZ = (XYZ * 100).astype(np.uint8)
    XYZ = XYZ.reshape(og_shape)

    return XYZ

def rgb_to_hsv(rgb):
    og_shape = rgb.shape
    rgb = rgb.reshape(-1, 3) / 255.0

    R = rgb[:, 0]
    G = rgb[:, 1]
    B = rgb[:, 2]

    
    Cmax = np.max(rgb,axis=1)


    Cmin = np.min(rgb,axis=1)


    delta = Cmax - Cmin



    H = np.zeros_like(R)

    V = Cmax

    S = np.where(Cmax == 0.0, 0, delta / Cmax)

    mask_r = (delta != 0.0) & (Cmax == R)
    mask_g = (delta != 0.0) & (Cmax == G)
    mask_b = (delta != 0.0) & (Cmax == B)

    H = np.where(mask_r, 60 * (((G - B) / delta) % 6), H)
    

    H = np.where(mask_g, 60 * (((B - R) / delta) + 2), H)
    

    H = np.where(mask_b, 60 * (((R - G) / delta) + 4), H)

    H = (H + 360) % 360

    H_scaled = H / 360.0 * 255.0
    S_scaled = S * 255.0
    V_scaled = V * 255.0
    
    HSV = np.stack([H_scaled, S_scaled, V_scaled], axis=-1)


    HSV = (HSV).astype(np.uint8)
    HSV = HSV.reshape(og_shape)

    return HSV