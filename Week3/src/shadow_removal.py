import numpy as np
import cv2
import matplotlib.pyplot as plt

def shadow_removal(img, threshold):
    # Ensure the image is not None
    if img is None:
        return None

    # --- Changes for Bottom-to-Top Search ---
    
    # 1. Start search seed near the bottom (instead of right)
    # Start from 3 rows up from the bottom edge
    search_seed = img.shape[0] - 7 
    
    # 2. Use a fixed column (e.g., the center column) for the search (instead of a fixed row)
    mid_width = img.shape[1] // 2 
    
    # Check if the search seed is valid
    if search_seed < 0:
        return img # Image is too small

    # Convert to float for stable gradient calculation
    prev_value = img[search_seed, mid_width].astype(np.float32)
    
    search = True
    counter = 0
    
    # Search from bottom (search_seed) upwards
    while search and search_seed >= 0:
        search_seed = search_seed - 1 
        
        if search_seed < 0:
            break
            
        # Access the pixel at the new row (search_seed) and fixed column (mid_width)
        value = img[search_seed, mid_width].astype(np.float32)

        # Calculate gradient (mean squared difference across channels)
        grad = np.mean((value - prev_value))

        if grad > threshold:
            search = False 

        prev_value = value
        counter += 1

    # Check for shadow existence based on counter range
    # The relevant dimension is now the height of the image
    img_height = img.shape[0]
    shadow = False
    if (counter > img_height / 100) and (counter < img_height / 100 * 30):
        shadow = True

    # If a shadow is detected, return the cropped image
    if shadow:
        # The transition point is at row: img.shape[0] - counter 
        # Crop from the START of the image down to this row.
        # This removes the assumed shadow region at the bottom.
        crop_row_end = img.shape[0] - counter
        return img[:crop_row_end, :] # All columns, up to crop_row_end row
    
    return img

