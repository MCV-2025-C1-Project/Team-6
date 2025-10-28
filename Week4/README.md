# Week 3

## Introduction

In **Week 3**, the system extends the Week 2 pipeline with **noise removal,  **feature extraction** using **frequency-based and texture-based descriptors* and spliting techinques** to improve image retrieval robustness.  

### Main Improvements
1. **Noise Filtering:**  
   Multiple denoising techniques (mean, median, Gaussian, bilateral, adaptive median) are implemented and evaluated to enhance descriptor stability.
2. **Shadow Removal:**  
   A custom algorithm removes border shadows adaptively based on image intensity gradients and direction analysis.
3. **Improved Background Removal:**  
   A more reliable morphological gradient method is used to segment paintings while suppressing image borders and fitting precise cropping polygons.
4. **Multiple Descriptors:**  
   - **DCT Descriptors:** Frequency-domain representation of spatial crops.  
   - **LBP Descriptors:** Texture-based representation for robustness to illumination.
5. **Two paintings detection and Spliting algorithm:**  
   A techinque that analayzies Conex Components of the dilated contour image to identify when there are two paintings or one, in order to split them to do the retrival separetdly. 

---

## Reproducing Results for Task 6

To reproduce the results for **Task 5**, follow these steps:

1. Ensure the image database is available at the root of the project: `Team-6/BBDD`.
2. Download the test sets for Week 3 (`qst1_w3` and `qst2_w3`). **Optional:** place them in `Team-6/Week3`.
3. Run the main script:
```bash
python main.py
```
By default, the script looks for the test sets in the directory `Team-6/Week3`. If your test set is located elsewhere, you can specify a custom path using `-dir1` and `-dir2`:
```bash
python main.py -dir1 PATH_TO_TEST_SET1 -dir2 PATH_TO_TEST_SET2
```
> **Note:** The arguments are optional and only required if your test sets are **not** in the default location (`Team-6/Week3`).

---

## Scripts Overview

This section provides a brief description of each script in the `src` folder:

- **`main.py`**  
  Main entry point for test set retrieval (`qst1_w3`, `qst2_w3`). Handles denoising, spliting, background removal, shadow removal, descriptor computation, and retrieval for the pickle files.

- **`main_backg.py`**  
  Evaluates background segmentation and splitting algorithm. Computes precision, recall, and F1-score with and without denoising.

- **`background_remover.py`**  
  Implements morphological gradient-based background removal. Detects polygon corners for accurate cropping and suppresses borders for robust segmentation.

- **`bg_noise_fliter.py`**  
  Comprehensive noise filtering module implementing mean, median, Gaussian, bilateral, and adaptive median filters. Used to preprocess the images before doing the spliting and the background segmentation.

- **`filter_noise.py`**  
  Alternative noise filtering script with core denoising functions and adaptive median filter implementation.

- **`shadow_removal.py`**  
  Removes shadows near image borders using gradient-based directional search. Ensures clean crops before descriptor computation.

- **`image_split.py`**  
  Automatically detects and separates two paintings in a single image using component analysis on edge masks.

- **`dct_descriptors.py`**  
  Computes frequency-based descriptors using the Discrete Cosine Transform (DCT). Supports grayscale and hybrid RGB-HS modes, spatial crops, and background removal.

- **`lbp_descriptors.py`**  
  Computes texture-based Local Binary Pattern (LBP) descriptors using `skimage.feature`. Supports simple and multiscale LBP.

- **`params.py`**  
Configuration for experiments and best configurations discovered during them.

- **`main_task4.py`**  
Main entry point for the development set retrieval (`qsd1_w3`, `qsd2_w3`). Handles denoising, spliting, background removal, shadow removal, descriptor computation, and retrieval evaluation.

Moreover, there are two additional folders:

- **`evaluations`**  
  Functions to compute metrics and distances.

- **`utils`**  
  Helpers for color conversion, histogram computation, I/O operations, and visualization.



