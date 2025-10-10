# Week 2

## Contents
1. [Introduction](#introducion)
2. [Project Structure and overview](#project-structure-and-overview)
3. [Database](#database)
4. [Run the program](#run-the-program)

## [Introduction](#introduction)
In **Week 2**, we build upon the Week 1 solution with two key upgrades:

1. **Spatial descriptors** are added on top of color-based descriptors to capture spatial layout information of each painting.
2. **Background segmentation** models the background color distribution so the descriptor focuses primarily on the painting.

These changes aim to improve retrieval robustness and the semantic relevance of descriptors.


## [Project Structure and overview](#project-structure-and-overview)
The main code lives in the `src/` directory. Below is a brief description of each component:

### Core Scripts

- **`background.py`**  
  Methods for background segmentation and utilities to evaluate segmentation quality.

- **`evaluate_crop_descriptors.py`**  
  Evaluates retrieval by experimenting with cropped query images: loads precomputed BBDD descriptors, computes query descriptors, and ranks results with multiple distance metrics.

- **`evaluate_piramid_descriptors.py`**  
  Evaluates retrieval using spatial *pyramid* descriptors.

- **`params.py`**  
  Experiments and best configurations discovered during testing.

- **`piramid_descriptors.py`**  
  Computes spatial (pyramid-based) descriptors from color histograms at multiple spatial levels.

Moreover there is two extra folders `evaluations` and `utils`:

- **`evaluations/`**  
  Functions to compute metrics, distances, and overall evaluation results.

- **`utils/`**  
  Helpers for color-space conversions, histogram computation/plotting, I/O, and visualization.

Finally, the code is orchestarted by **`main.py`**:
- **`main.py`**  
  Main entry point for result generation. It handles data loading, background segmentation, descriptor extraction, similarity computation, and evaluation of predictions.

## [Database](#database)
This Week's project uses two main databases: 
1) **QSD1_W2** Used for evaluating the query retreival system from Task 1.
2) **QSD2_W2** Used for evaluating the query retrieval system + segmentation pipeline.

## [Run the program](#run-the-program)

To reproduce the results for Task 6, follow these steps:
1. Download the test sets `qst1_w2` and `qst2_w2`. **Optional** place it in `Team-6/Week2/qst1_w2/` and `Team-6/Week2/qst1_w2/`, respectively.
2. Run the main script:
```bash
python main.py
```

By default, the script looks for the test set in the path `Team-6/Week2/qst1_w2/`. If your test set is located elsewhere, you can specify a custom path using `-d`:
```bash
python main.py -d PATH_TO_TEST_SET
```
> **Note:** The `-d` argument is optional and only required if your test set is **not** in the default location (`Team-6/Week1/qst1_w1/`).

