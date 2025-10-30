import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from utils.io_utils import read_images, write_pickle


SCRIPT_DIR = Path(__file__).resolve().parent

Keypoints = List[cv.KeyPoint]
Descriptors = Optional[np.ndarray]

#For saving keypoints
import cv2

def serialize_keypoints(kps):
    # kps: List[cv2.KeyPoint] -> List[tuple]
    return [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in kps]

def deserialize_keypoints(serial):
    # serial: List[tuple] -> List[cv2.KeyPoint]
    out = []
    for x, y, size, angle, response, octave, class_id in serial:
        out.append(cv2.KeyPoint(x=float(x), y=float(y),
                                size=float(size), angle=float(angle),
                                response=float(response), octave=int(octave),
                                class_id=int(class_id)))
    return out

def serialize_keypoints_list(list_of_kps):
    # List[List[cv2.KeyPoint]] -> List[List[tuple]]
    return [serialize_keypoints(kps) for kps in list_of_kps]

def deserialize_keypoints_list(list_of_serial):
    # List[List[tuple]] -> List[List[cv2.KeyPoint]]
    return [deserialize_keypoints(s) for s in list_of_serial]


# Helper 
def _descriptor_len_and_dtype(method: str) -> Tuple[int, np.dtype]:
    m = method.lower()
    if m == "orb":
        return cv.ORB_create().descriptorSize(), np.uint8   # 32, uint8
    # sift or hsift
    return cv.SIFT_create().descriptorSize(), np.float32

### Extractors ###
class SIFTExtractor:
    """Reusable SIFT wrapper. Single thread. Uses DoG and gradient histograms."""
    def __init__(self):
        self.sift = cv.SIFT_create()

    def detect(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Keypoints:
        """Detect keypoints."""
        return self.sift.detect(gray, mask)

    def compute(self, gray: np.ndarray, keypoints: Keypoints) -> Tuple[Keypoints, Descriptors]:
        """Compute descriptors for given keypoints."""
        return self.sift.compute(gray, keypoints)

    def detect_and_compute(
        self, gray: np.ndarray, mask: Optional[np.ndarray] = None
        ) -> Tuple[Keypoints, Descriptors]:
        """Detect keypoints and compute descriptors in one pass."""
        return self.sift.detectAndCompute(gray, mask)
    
class ORBExtractor:
    """Reusable ORB wrapper. Single thread. Uses FAST (kp) and BRIEF (descriptors)."""
    def __init__(self):
        self.orb = cv.ORB_create()

    def detect(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Keypoints:
        """Detect keypoints."""
        return self.orb.detect(gray, mask)

    def compute(self, gray: np.ndarray, keypoints: Keypoints) -> Tuple[Keypoints, Descriptors]:
        """Compute descriptors for given keypoints."""
        return self.orb.compute(gray, keypoints)

    def detect_and_compute(
        self, gray: np.ndarray, mask: Optional[np.ndarray] = None
        ) -> Tuple[Keypoints, Descriptors]:
        """Detect keypoints and compute descriptors in one pass."""
        return self.orb.detectAndCompute(gray, mask)
    
class HarrisSIFTExtractor:
    def __init__(
        self,
        block_size: int = 3,          
        ksize: int = 3,               
        k: float = 0.04,            
        max_corners: int = 2000,
        quality_level: float = 0.01,  
        min_distance: float = 8.0,    
        kp_size: float = 9.0          
    ):
        self.block_size = int(block_size)
        self.ksize = int(ksize)
        self.k = float(k)
        self.max_corners = int(max_corners)
        self.quality_level = float(quality_level)
        self.min_distance = float(min_distance)
        self.kp_size = float(kp_size)

        self.sift = cv.SIFT_create()
    
    def detect(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Keypoints:
        """Detect Harris corners and convert them to cv.KeyPoint list."""
        corners = cv.goodFeaturesToTrack(
            image=gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=self.block_size,
            useHarrisDetector=True,
            k=self.k,
        )

        kps: Keypoints = []
        if corners is None:
            return kps

        # Convert each corner to a KeyPoint.
        for c in corners:
            x, y = float(c[0, 0]), float(c[0, 1])
            kps.append(cv.KeyPoint(x=x, y=y, size=self.kp_size))
        return kps
    
    def compute(self, gray: np.ndarray, keypoints: Keypoints) -> Tuple[Keypoints, Descriptors]:
        """Compute SIFT descriptors on provided keypoints."""
        return self.sift.compute(gray, keypoints)

    def detect_and_compute(
        self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[Keypoints, Descriptors]:
        """Detect Harris corners, then compute SIFT descriptors."""
        kps = self.detect(gray, mask)
        if not kps:
            return kps, None
        return self.compute(gray, kps)


### Helpers ###
def compute_one_descriptor(gray: np.ndarray, method: str,
                 sift: SIFTExtractor, orb: ORBExtractor, hsift: HarrisSIFTExtractor
                ) -> Tuple[List[cv.KeyPoint], Optional[np.ndarray]]:
    m = method.lower()
    if m == "sift":
        return sift.detect_and_compute(gray)
    if m == "orb":
        return orb.detect_and_compute(gray)
    if m == "hsift":
        return hsift.detect_and_compute(gray)
    raise ValueError(f"Unknown method {method}")

### Main method ###

# returns 3D matrix with size (keypoints, descriptor, image)
def compute_descriptors(
        imgs: List[np.ndarray], 
        method='sift',
        save_pkl=False
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    print("Computing descriptors...")
    sift = SIFTExtractor()
    orb = ORBExtractor()
    hsift = HarrisSIFTExtractor()

    desc_len, want_dtype = _descriptor_len_and_dtype(method)

    descriptors = []
    keypoints = [] # not necessary, just to see results in main method of the script
    for img in imgs:
        # convert to gray uint8
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) if img.ndim == 3 else img
        if gray.dtype != np.uint8:
            gray = np.clip(gray*(255 if gray.dtype.kind=='f' and gray.max()<=1 else 1),0,255).astype(np.uint8)
        
        kps, des = compute_one_descriptor(gray, method, sift, orb, hsift)
        if des is None:
            des_out = np.empty((0, desc_len), dtype=want_dtype)
        else:
            # SIFT/HSIFT -> float32 ; ORB -> uint8
            des_out = np.ascontiguousarray(des.astype(want_dtype, copy=False))

        keypoints.append(kps)
        descriptors.append(des_out)

    if save_pkl:
        # Make directory if not setted up
        os.makedirs(SCRIPT_DIR / "descriptors", exist_ok=True)
        os.makedirs(SCRIPT_DIR / "keypoints", exist_ok=True)
        
        print("Saving descriptors and keypoints...")
        write_pickle(descriptors, SCRIPT_DIR / "descriptors" / f"descriptors_{method}.pkl")
        # SERIALIZA antes de guardar
        write_pickle(serialize_keypoints_list(keypoints), SCRIPT_DIR / "keypoints" / f"keypoints_{method}.pkl")
    return keypoints, descriptors

if __name__=="__main__":
    imgs = read_images(SCRIPT_DIR.parent / 'qsd1_w4')
    keys, desc = compute_descriptors(imgs, 'orb', save_pkl=False)

    i = 0
    for img, k in zip(imgs, keys):
        gray= cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        img_orb=cv.drawKeypoints(gray,k,img)
        cv.imwrite(f'test_{i}.jpg',img_orb)
        i += 1

