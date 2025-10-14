# Week 2

## Introduction
In **Week 2**, we build upon the Week 1 solution with two key upgrades:

1. **Spatial descriptors** are added on top of color-based descriptors to capture spatial layout information of each painting.
2. **Background segmentation** removes the background from its color so that the retrieval only works with paintings.

These changes aim to improve retrieval robustness and the semantic relevance of the descriptors.

## Reproducing results for Task 6

To reproduce the results for **Task 6**, follow these steps:

1. Ensure the image database is available at the root of the project: `Team-6/BBDD`.
2. Download the test sets for Week 2 (`qst1_w2` and `qst2_w2`). **Optional:** place them in `Team-6/Week2`.
3. Run the main script:
```bash
python main.py
```
By default, the script looks for the test sets in the directory `Team-6/Week2`. If your test set is located elsewhere, you can specify a custom path using `-dir1` and `-dir2`:
```bash
python main.py -dir1 PATH_TO_TEST_SET1 -dir2 PATH_TO_TEST_SET2
```
> **Note:** The arguments are optional and only required if your test sets are **not** in the default location (`Team-6/Week2`).

## Scripts Overview

This section provides a brief description of each script in the `src` folder:

- **`background.py`**  
  Methods for background segmentation and utilities to evaluate segmentation quality.

- **`descriptors.py`**  
  Computes spatial (pyramid or only grid based) descriptors from color histograms at multiple spatial levels.

- **`development_main.py`**  
  Result generation as `main.py` but for the development sets (with MAP@k computation).

- **`evaluate_crop_descriptors.py`**  
  Evaluates retrieval by experimenting with cropped query images: loads precomputed BBDD descriptors, computes query descriptors, and ranks results with multiple distance metrics.

- **`evaluate_piramid_descriptors.py`**  
  Evaluates retrieval using spatial *pyramid* descriptors.

- **`main.py`**  
  Main entry point for result generation for the test sets. It handles data loading, background segmentation, descriptors extraction, similarity computation, and evaluation of predictions.

- **`params.py`**  
  Configuration for experiments and best configurations discovered during them.

Moreover, there are two additional folders:

- **`evaluations`**  
  Functions to compute metrics and distances.

- **`utils`**  
  Helpers for color conversion, histogram computation, I/O operations, and visualization.
