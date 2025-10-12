# Week 1

## Reproducing results for Task 4

To reproduce the results for **Task 4**, follow these steps:

1. Ensure the image database is available at the root of the project: `Team-6/BBDD`.
2. Download the test set for Week 1 (`qst1_w1`). **Optional:** place it in `Team-6/Week1/qst1_w1/`.
3. Run the main script:
```bash
python main.py
```
By default, the script looks for the test set in the path `Team-6/Week1/qst1_w1/`. If your test set is located elsewhere, you can specify a custom path using `-d`:
```bash
python main.py -d PATH_TO_TEST_SET
```
> **Note:** The `-d` argument is optional and only required if your test set is **not** in the default location (`Team-6/Week1/qst1_w1/`).


## Scripts Overview

This section provides a brief description of each script in the `src` folder:

- **`main.py`**  
  Main entry point for result generation. It handles data loading, descriptor extraction, similarity computation, and evaluation of predictions.

- **`color_spaces.py`**  
  Functions to convert images between different color spaces (e.g., RGB, BGR, HSV...), enabling experiments with various image representations.

- **`descriptors.py`**  
  Computes image descriptors based on histograms.

- **`evaluate_descriptors.py`**  
  Evaluates and compares descriptor methods and distance metrics using MAP@k.

- **`histogram.py`**  
  Functions for computing and manipulating histograms.

- **`io_utils.py`**  
  Utilities for reading images and reading/writing data using pickle files.

- **`metrics.py`**  
  Implements the MAP@k (Mean Average Precision at k) metric used for evaluating retrieval performance.

- **`params.py`**  
  Contains configuration parameters for grid search experiments and the setup for the best-performing methods.

- **`plots.py`**  
  Functions for visualizing descriptor comparisons and prediction outputs.

- **`similarity_measures.py`**  
  Implements various similarity/distance measures used to compare image descriptors.
