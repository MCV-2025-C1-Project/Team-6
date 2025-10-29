from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from utils.io_utils import read_images, read_pickle


SCRIPT_DIR = Path(__file__).resolve().parent

Keypoints = List[cv.KeyPoint]
Descriptors = Optional[np.ndarray]

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
        method='sift'
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    
    sift = SIFTExtractor()
    orb = ORBExtractor()
    hsift = HarrisSIFTExtractor()

    descriptors = []
    keypoints = [] # not necessary, just to see results in main method of the script
    for img in imgs:
        # convert to gray uint8
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if gray.dtype != np.uint8:
            gray = np.clip(gray*(255 if gray.dtype.kind=='f' and gray.max()<=1 else 1),0,255).astype(np.uint8)
        key, des = compute_one_descriptor(gray, method, sift, orb, hsift)
        descriptors.append(des.astype(np.float32) if des is not None else np.empty((0, cv.SIFT_create().descriptorSize() if method!="orb" else cv.ORB_create().descriptorSize()), np.float32))
        keypoints.append(key)
    return keypoints, descriptors


    
if __name__=="__main__":
    # sift = SIFTExtractor()
    # orb = ORBExtractor()
    # hsift = HarrisSIFTExtractor()
    # img = cv.imread(SCRIPT_DIR.parent / 'qsd1_w4' / '00001.jpg')
    # gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # kp_sift, des_sift = sift.detect_and_compute(gray)
    # kp_orb, des_orb = orb.detect_and_compute(gray)
    # kp_h, des_h = hsift.detect_and_compute(gray)
    # img_sift=cv.drawKeypoints(gray,kp_sift,img) #,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # img_orb=cv.drawKeypoints(gray,kp_orb,img) #,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # img_h=cv.drawKeypoints(gray,kp_h,img)
    # print(des_sift)
    # print(len(kp_sift))
    # print(len(des_sift))
    # print(des_sift.shape)
    # cv.imwrite('sift_keypoints.jpg',img)
    # cv.imwrite('orb_keypoints.jpg',img)
    # cv.imwrite('h_keypoints.jpg',img)
    # print(read_pickle(SCRIPT_DIR.parent / 'qsd1_w4' / 'gt_corresps.pkl'))

    imgs = read_images(SCRIPT_DIR.parent / 'qsd1_w4')
    keys, desc = compute_descriptors(imgs, 'sift')

    i = 0
    for img, k in zip(imgs, keys):
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_orb=cv.drawKeypoints(gray,k,img)
        cv.imwrite(f'test_{i}.jpg',img_orb)
        i += 1
