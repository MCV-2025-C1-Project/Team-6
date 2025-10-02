from typing import List
import numpy as np

# TODO: Create the descriptors. It can be extended and can be done by using histogram.py

def compute_descriptors(imgs: List[np.ndarray], method: str = "rgb") -> List[np.ndarray]:
    # TODO: Improve the explanation, returns the descriptors for the images.
    if method == "rgb":
        return [np.zeros(10)]

    elif method == "hsv":
        return [np.zeros(10)]

    else:
        raise ValueError(f"Invalid method ({method}) for computing image descriptors!")
    