import argparse
from pathlib import Path

from utils.io_utils import read_images
from src.background import find_best_mask, apply_best_method_and_plot
from src.params import segmentation_experiments, best_config_segmentation

SCRIPT_DIR = Path(__file__).resolve().parent


def main(data_dir: Path) -> None:

    # Read query images to remove background (just .jpg)
    images = read_images(data_dir)
    
    # Read ground truths (extension is .png)
    masks = read_images(data_dir, extension="png")

    # 'Grid Search' - Find best method (already found!)
    # best_solution = find_best_mask(images, masks, segmentation_experiments)

    # Find masks and plot with best method
    _ = apply_best_method_and_plot(
        images,
        best_config_segmentation)

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