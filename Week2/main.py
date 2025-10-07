import argparse

from pathlib import Path
from io_utils import read_images
from background import remove_background

SCRIPT_DIR = Path(__file__).resolve().parent

def main(data_dir: Path) -> None:

    # Read query images to remove background
    images = read_images(data_dir)
    
    # Read ground truths (extension is .png)
    ground_truth = read_images(data_dir, extension="png")

    # FUNCTION OF REMOVE BACKGROUND
    remove_background(images)

    # TODO: EVALUATE THE SEGMENTATION WITH THE GROUND TRUTH


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