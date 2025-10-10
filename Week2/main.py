import argparse
import os
from pathlib import Path

import numpy as np

from utils.plots import plot_query_results
from evaluations.metrics import mean_average_precision
from evaluations.similarity_measures import compute_similarities
from src.piramid_descriptors import compute_spatial_descriptors
from utils.io_utils import read_images, write_pickle, read_pickle
from src.background import find_best_mask, apply_best_method_and_plot
from src.params import segmentation_experiments, best_config_segmentation, best_config_descriptors

SCRIPT_DIR = Path(__file__).resolve().parent

def main(data_dir: Path) -> None:

    # Load ground truth
    gt = read_pickle(data_dir / "gt_corresps.pkl")

    # Read query images to remove background (just .jpg)
    images = read_images(data_dir)
    
    # Read ground truths (extension is .png)
    masks = read_images(data_dir, extension="png")

    # 'Grid Search' - Find best method (already found!)
    # best_solution = find_best_mask(images, masks, segmentation_experiments)

    # Find masks and plot with best method
    masks = apply_best_method_and_plot(
        images,
        best_config_segmentation)
    
    #Apply masks to images
    masked_images = [img * mask[:, :, None] for img, mask in zip(images, masks)]

    # Descriptor parameters
    method = best_config_descriptors

    # Load precomputed descriptors for the BBDD
    bbdd_descriptors = read_pickle(Path(__file__).resolve().parent / "descriptors" / "hsv_16_[1, 10, 20]_pyramid_descriptors.pkl")    

    #TODO: FALTA QUE EN EL CALCULO DE DESCRIPTORES, NO SE CUENTEN LOS PIXELES NEGROS DE MASK                                                                           

    # Compute query images descriptors
    descriptors = compute_spatial_descriptors(masked_images, method=method["color_space"], n_bins=method["n_bins"], pyramid=True, pyramid_levels=method["pyramid_levels"], n_crops=method["n_crops"])

    # Compute similarities
    similarities = compute_similarities(descriptors, bbdd_descriptors, method["metric"])

    # Sort the indices resulting from the similarities sorting
    indices = np.argsort(similarities, axis=1)

    # Save results for k=10
    k = 10
    results = indices[:, :k].astype(int).tolist()
    write_pickle(results, SCRIPT_DIR / "result.pkl")


    # Compute MAP score
    map_score = mean_average_precision(indices, gt, k)
    print(f"MAP@{k} score: {map_score:.4f}")

    # Make directory if not setted up
    os.makedirs(SCRIPT_DIR / "results", exist_ok=True)

    plot_query_results(queries=masked_images, 
                       results=indices[:, :k], 
                       similarity_values=np.take_along_axis(similarities, indices[:, :k], axis=1), 
                       k=k, 
                       save_path=SCRIPT_DIR / "results" / "query_results.png")



    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir',
        type=Path,
        default=SCRIPT_DIR / "qsd2_w2" / "qsd2_w1",  
        help='Path to the dataset directory.'
    )
    data_dir = parser.parse_args().data_dir

    # Check directory
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory.")

    main(data_dir)