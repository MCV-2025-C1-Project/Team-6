import pickle
from typing import Any, List
import cv2
import numpy as np
from pathlib import Path

# Read data from pickle file
def read_pickle(file_path: Path) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


# Write data to a pickle file
def write_pickle(data: Any, file_path: Path) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def read_images(dir_path: Path) -> List[np.ndarray]:
    """
    Reads all JPG images from the given directory using OpenCV (sorted by their filenames).

    Args:
        dir_path (Path):    Path to the image directory.

    Returns:
        List[np.ndarray]:   A list containing the images.
    """
    return [
        cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        for img_path in sorted(dir_path.glob("*.jpg"))
    ]
