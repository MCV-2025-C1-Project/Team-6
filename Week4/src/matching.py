from pathlib import Path

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from descriptors import compute_one_descriptor, ORBExtractor, SIFTExtractor, HarrisSIFTExtractor
from params import FLANN_MATCHER_PARAMS
"""
information extracted from: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
"""
### Matching methods ###
# Those method already implements Lowe's ratio and counts the 'survivors'
# Need to experiment with bidirectional consistency or RANSAC
# TODO: Oriol, need to finish the loew's ratio so it is beautiful

class ORBMatcher:
    def __init__(self, norm=cv.NORM_HAMMING, cross_check=True):
        self.bf = cv.BFMatcher(normType=norm, crossCheck=cross_check)

    def brute_force_match(self, des1, des2, top_k=None):
        if des1 is None or des2 is None or len(des1)==0 or len(des2)==0:
            print("No descriptors to match.")
            return []
        matches = self.bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        return matches[:top_k] if top_k else matches

class SIFTMatcher:
    def __init__(self, norm=cv.NORM_L2, crossCheck=False):
        self.bf = cv.BFMatcher(norm, crossCheck)

    def brute_force_match(self, des1, des2, ratio_test = 0.75):
        if des1 is None or des2 is None or len(des1)==0 or len(des2)==0:
            print("No descriptors to match.")
            return []
        knn = self.bf.knnMatch(des1,des2,k=2)
        good = []
        for neigh in knn:
            if len(neigh) < 2:   # skip if no second neighbor
                continue
            m, n = neigh[0], neigh[1]
            if m.distance < ratio_test * n.distance:
                good.append([m]) 
        return good
 
class FLANNMatcher:
    def __init__(self, method):
        assert method in ['orb', 'sift']
        self.method = method
        self.flann = cv.FlannBasedMatcher(
            FLANN_MATCHER_PARAMS["index_params"][f"{method}"],
            FLANN_MATCHER_PARAMS["SearchParams"])
        
    def match(self, des1, des2, top_k=2, ratio_test=0.7):
        if des1 is None or des2 is None or len(des1)==0 or len(des2)==0:
            print("No descriptors to match.")
            return []
        if self.method == 'sift' and des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
        if self.method == 'sift' and des2.dtype != np.float32:
            des2 = des2.astype(np.float32)

        knn = self.flann.knnMatch(des1,des2,k=top_k)
        good = []
        for neigh in knn:
            if len(neigh) < 2:
                continue
            m, n = neigh[0], neigh[1]
            if m.distance < ratio_test * n.distance:
                good.append([m]) 
        return good

if __name__=="__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    img1 = cv.imread(SCRIPT_DIR.parent / 'qsd1_w4' / '00001.jpg')
    img2 = cv.imread(SCRIPT_DIR.parent / 'qsd1_w4' / '00002.jpg')
    gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    matcher = ORBMatcher()
    orb = ORBExtractor()
    sift = SIFTExtractor()
    harris = HarrisSIFTExtractor()

    kp1, des1 = compute_one_descriptor(gray1, method='orb',sift=sift,orb=orb, hsift=harris)
    kp2, des2 = compute_one_descriptor(gray2, method='orb',sift=sift,orb=orb, hsift=harris)
    
    matches = matcher.brute_force_match(des1, des2, top_k = 50)

    vis = cv.drawMatches(img1, kp1, img2, kp2, matches, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()












