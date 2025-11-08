# Week 4 – SIFT Feature Extraction and Inlier-Based Retrieval

## Introduction

In **Week 4**, the system builds upon the image preprocessing pipeline from previous weeks and introduces a **local feature–based retrieval framework** using **SIFT and ORB descriptors**.  
The goal is to improve retrieval robustness by detecting stable keypoints, computing discriminative descriptors, and verifying geometric consistency between query and database images.

### Main Improvements

1. **SIFT Keypoint Detection and Descriptor Extraction**  
   The system detects scale and rotation-invariant features using the **Difference of Gaussians (DoG)** method and computes **SIFT descriptors** to represent local texture and shape patterns robustly.

2. **Geometric Verification with RANSAC**  
   After matching keypoints, **homography estimation** with **RANSAC** is applied to filter out spurious correspondences and keep only **inliers** — consistent matches under projective transformations.

3. **Mutual Matching and Ratio Test**  
   The retrieval pipeline employs **bidirectional matching** and **Lowe’s ratio test** to improve matching precision and reject ambiguous correspondences.

4. **Grid Search for Parameter Optimization**  
   A dedicated script (`grid_search.py`) systematically explores combinations of SIFT parameters  
   (e.g., inlier thresholds, ratio test thresholds, reprojection errors) to maximize retrieval accuracy and mean Average Precision (mAP@2).

5. **Automatic Handling of Multiple Paintings**  
   The retrieval logic supports **multi-painting detection** by inferring from the number of inliers whether a query contains one or two artworks.  
   Separate retrieval is performed per detected painting region.

6. **Evaluation with Ground Truth Correspondences**  
   The final predictions are compared against `gt_corresps.pkl` to compute metrics such as **accuracy**, **mean precision**, and **mAP@2**, allowing objective performance comparison across descriptor configurations.

---
## Reproducing Results for Task 3
To reproduce the results for **Task 3**, follow these steps:

1. Ensure the image database is available at the root of the project: `Team-6/BBDD`.
2. Download the test sets for Week 3 (`qst1_w4`). **Optional:** place them in `Team-6/Week4`.
3. Run the main script:
```bash
python main.py
```
By default, the script looks for the test sets in the directory `Team-6/Week4`. If your test set is located elsewhere, you can specify a custom path using `-dir1`:
```bash
python main.py -dir1 PATH_TO_TEST_SET1
```
> **Note:** The arguments are optional and only required if your test sets are **not** in the default location (`Team-6/Week4`).

## Scripts Overview

This section provides a brief description of each script in the `src` folder:

- **`main.py`**  
  Main entry point for test set retrieval (`qst1_w4`). Handles denoising, spliting, background removal, shadow removal, keypoint search, descriptor computation, and retrieval for the pickle files.

- **`background_remover.py`**  
  Implements morphological gradient-based background removal. Detects polygon corners for accurate cropping and suppresses borders for robust segmentation.

- **`bg_noise_fliter.py`**  
  Comprehensive noise filtering module implementing mean, median, Gaussian, bilateral, and adaptive median filters. Used to preprocess the images before doing the spliting and the background segmentation.

- **`shadow_removal.py`**  
  Removes shadows near image borders using gradient-based directional search. Ensures clean crops before descriptor computation.

- **`image_split.py`**  
  Automatically detects and separates two paintings in a single image using component analysis on edge masks.

- **`descriptors.py`**  
  Computes local feature–based descriptors using Harris + SIFT, SIFT and ORB.

- **`params.py`**  
  Configuration for best configurations discovered during experiments.

- **`gs_configs.py`**  
  Configurations used for testing experiments.

- **`grid_search.py`**  
  Script used to perform repetitive experiments and test them out, suign different descriptor modes and 
  different parameters.

- **`matching.py`**  
  Script used to find correspondences between the query image and images in the database, implementing ratio test and bidirectional ratio test.

-**`scoring.py`**  
  Evaluate how good the matches between the query and the database images are.
  After obtaining correspondances, assign each database image a score that reflects how strongly it matches the query. Use scoring by length and scoring by inliers.


Moreover, there are two additional folders:

- **`evaluations`**  
  Functions to compute metrics and distances.

- **`utils`**  
  Helpers for color conversion, histogram computation, I/O operations, and visualization.

- **`descriptors`**  
  Cache folder to save descriptors of the BBDD computed with a certain method.

- **`keypoints`**  
  Cache folder to save keypoints of the BBDD computed with a certain method.

