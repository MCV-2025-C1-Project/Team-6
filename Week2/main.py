import argparse
from pathlib import Path

from io_utils import read_images
from background import find_best_mask, apply_best_method_and_plot

SCRIPT_DIR = Path(__file__).resolve().parent


def main(data_dir: Path) -> None:

    # Read query images to remove background (just .jpg)
    images = read_images(data_dir)
    
    # Read ground truths (extension is .png)
    masks = read_images(data_dir, extension="png")

    # 'Grid Search' - Find best method
    best_solution = find_best_mask(images, masks)

    # Find masks and plot with best method
    masks = apply_best_method_and_plot(
        images,
        best_solution["best_method"])

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