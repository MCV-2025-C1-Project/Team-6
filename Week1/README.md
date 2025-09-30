# Week 1 

### Tasks
1. **Compute image descriptors**  
   - Color histograms (1D, gray level or color spaces: RGB, HSV, etc.)

2. **Implement similarity measures**  
   - Euclidean, L1, χ², Histogram intersection, Hellinger kernel…

3. **Build retrieval system**  
   - For each query image (QSD1), compute similarities to museum dataset  
   - Return top K results (evaluate with mAP@K)

4. **Blind test (QST1)**  
   - Submit top results for each query (list of lists, pkl format)  
   - Best two methods
     

## Quick Setup

Clone the repository

```bash
git clone git@github.com:MCV-2025-C1-Project/Team-6.git
cd Team-6
```

Create the environment

```bash
conda env create -f environment.yml
conda activate group6_env
```