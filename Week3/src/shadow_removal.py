import numpy as np
import cv2
import matplotlib.pyplot as plt

def shadow_removal(img, threshold,axises = [0], directions = [-1]):
    # Ensure the image is not None
    if img is None:
        return None
    for axis in axises:
        for direction in directions:
                
        
        
            # 1. Start search seed near the bottom (instead of right)
            # Start from 3 rows up from the bottom edge
            if direction == 1:
                search_seed = 6
            else:
                search_seed = img.shape[axis] - 6
            
            
            # 2. Use a fixed column (e.g., the center column) for the search (instead of a fixed row)
            height = img.shape[0] // 2 
            width = img.shape[1] // 4 
            if axis == 0:
                height = search_seed
            else:
                width = search_seed
            
            
            
            # Convert to float for stable gradient calculation
            prev_value = img[height, width].astype(np.float32)
            
            counter = 6

            search = True
            
            # Search from bottom (search_seed) upwards
            while search and height >= 0 and width >= 0:
                
                
                if axis == 0:
                    height = height + direction
                    if height == img.shape[0]:
                        break
                else:
                    width = width + direction
                    if width == img.shape[1]:
                        break
                
                
                if search_seed < 0:
                    break


                # Access the pixel at the new row (search_seed) and fixed column (mid_width)
                value = img[height, width].astype(np.float32)

                # Calculate gradient (mean squared difference across channels)
                grad = np.mean((value - prev_value))

                if grad > threshold:
                    search = False 

                prev_value = value
                counter += 1

            # Check for shadow existence based on counter range
            # The relevant dimension is now the height of the image
            img_height = img.shape[0]
            img_width = img.shape[1]
            
            shadow = False
            if axis == 0:
                if (counter > img_height / 100) and (counter < img_height / 100 * 30):
                    shadow = True
            else:
                if (counter > img_width / 100) and (counter < img_width / 100 * 30):
                    shadow = True

            # If a shadow is detected, return the cropped image
            if shadow:
                if axis == 0:
                    if direction == 1:
                        img = img[counter:,:]
                    else:
                        img = img[:img.shape[0] - counter,:]
                else:
                    if direction == 1:
                        img = img[:, counter:]
                    else:
                        img = img[:, :img.shape[1] - counter]
            
    return img

