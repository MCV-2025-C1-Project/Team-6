import argparse
import matplotlib.pyplot as plot
import numpy as np
from metrics import mean_average_precision
from pathlib import Path
from io_utils import read_images, read_pickle
from descriptors import compute_descriptors
from similarity_measures import compute_similarities

def main(data_dir: Path) -> None:
    # Read images
    images = read_images(data_dir)

    # Obtain database descriptors. TODO: Make this offline, reading from a .pkl file.
    # Size: [n_bbdd_imgs, descr_dim]
    # bbdd_descriptors = read_pickle("PATH_TO_THE_FILE")
    # TODO: THIS SHOULD BE DELETED!
    #import numpy as np
    #bbdd_descriptors = np.zeros(10)
    bbdd_descriptors = read_pickle("BBDD_2/BBDD_descriptors_rgb.pkl")

    # Compute query images descriptors
    # Size: [n_query_imgs, descr_dim]
    img_descriptors = compute_descriptors(images)

    # Compute similarities
    # Size: [n_query_imgs, n_bbdd_imgs]
    similarities = compute_similarities(img_descriptors, bbdd_descriptors['descriptors'])

    #print(similarities)


    # Choose the result sorting the similarities
    # # TODO: It is just sort each row and take the argmin for the indexes (the images from the BBDD will be loaded in order)
    results = np.sort(similarities, axis=1)

    #print(results.shape)
    #print("#############")
    #print(results)

    # # TODO: Save results in a pickle file called result (then rename it for each method used)
    for res in results:
        print(f" Most similar image to {res} in the BBDD is: {res[0]}")
        plot.imshow(images[res[0][1]])

    #predictions = [res[0][1] for res in results]
    predictions = [[t[1] for t in fila] for fila in results]

    #print(predictions)

    # # TODO: Use the metrics.py file to compute the MAP@K score
    gt = read_pickle(data_dir / "gt_corresps.pkl")
    print(gt)

    map_score = mean_average_precision(predictions, gt)

    print(map_score)

    # NOTE: WE CAN ADD MORE ARGUMENTS IN THE DATA PARSER TO ACCOUNT FOR THE 2 STRATEGIES TO USE, OR WE CAN MAKE 
    # COMPUTE DESCRIPTORS TO DO WHATEVER, THIS IS A FIRST SKELETON
    

if __name__ == "__main__":

    # Parse data directory argument (by default the dev set is used)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir',
        type=Path,
        default=Path(__file__).resolve().parent / "qsd1_w1",
        help='Path to the dataset directory.'
    )
    data_dir = parser.parse_args().data_dir

    # Check directory
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory.")

    # Process dataset
    main(data_dir)
