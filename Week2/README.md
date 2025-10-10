# Week 2

## Contents
1. [Introduction](#introducion)
2. [Project Structure and overview](#project-structure-and-overview)
3. [Database](#database)
4. [Run the program](#run-the-program)

## [Introduction](#introduction)
For this Week's implementation we upgrade Week's 1 solution with two main things:
1) First, we implement spatial descriptors moreover of color-based descriptors.
2) The painintgs are segmented via modelling the background color distribution, letting
the descriptor focus mainly on the painintg.

## [Project Structure and overview](#project-structure-and-overview)
This section provides a brief description of each script in the `src` folder:

**`background.py`**
Methods used for background segmentation and its evaluation.

**`evaluate_crop_descriptors.py`**
Evaluates retrieval on query images by loading precomputed BBDD descriptors, computing query descriptors and ranking with multiple metrics for crop-based descriptors.

**`evaluate_piramid_descriptors.py`**
Evaluates retrieval on query images by loading precomputed BBDD descriptors, computing query descriptors and ranking with multiple metrics for pyramid-based descriptors.

**`params.py`**
Contains experiments to test and the best configurations found.

**`piramid_descriptors.py`**
Computes spatial descriptors based on histograms.

Moreover there is two extra folders `evaluations` and `utils`:
1) **evaluations** contains all those scripts used for computing metrics and distances for evaluating the descriptors.
2) **utils** contains utils for converting between color spaces, compute and plot histograms, read and write files or plot results.

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

