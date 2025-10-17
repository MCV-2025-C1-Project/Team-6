import argparse
import numpy as np
import cv2
from pathlib import Path
from segmentation_eval import evaluation
from background_team2 import remove_background_morphological_gradient
from utils.io_utils import read_images
from image_split import split_image
from filter_noise import denoise_batch
from params import best_noise_params

SCRIPT_DIR = Path(__file__).resolve().parent

import matplotlib.pyplot as plt

def show_split_debug(image, parts, masks, mask_final):
    """
    Visualiza:
      - Original con línea de corte (si hay 2 partes)
      - Parte 1 + su máscara
      - Parte 2 + su máscara (si existe)
      - Máscara final reconstruida
    Y guarda un PNG en out_dir.
    """
    # ¿dónde se cortó? -> ancho de la parte izquierda
    cut_x = parts[0].shape[1] if len(parts) == 2 else None

    ncols = 4 if len(parts) == 2 else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))

    # 1) Original (con línea de corte si procede)
    axes[0].imshow(image)
    axes[0].set_title("Original")
    if cut_x is not None:
        axes[0].axvline(cut_x, linewidth=2)
    axes[0].axis("off")

    # 2) Parte 1 + máscara
    axes[1].imshow(parts[0])
    axes[1].imshow(masks[0], alpha=0.4)
    axes[1].set_title("Parte 1 + máscara")
    axes[1].axis("off")

    if len(parts) == 2:
        # 3) Parte 2 + máscara
        axes[2].imshow(parts[1])
        axes[2].imshow(masks[1], alpha=0.4)
        axes[2].set_title("Parte 2 + máscara")
        axes[2].axis("off")

        # 4) Máscara reconstruida
        axes[3].imshow(mask_final, cmap="gray")
        axes[3].set_title("Máscara reconstruida")
        axes[3].axis("off")
    else:
        # 3) Máscara final (cuando no hay split)
        axes[1].imshow(mask_final, cmap="gray")
        axes[1].set_title("Máscara final")
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()  


def main(data_dir: Path) -> None:

    # Read query images to remove background
    og_images = read_images(data_dir)

    images = denoise_batch(og_images, thresholds=best_noise_params)

    predictions = []
    ground_truths = []
    
    # Read ground truths (extension is .png)
    ground_truths = read_images(data_dir, extension="png")

    # Create output directory
    out_dir = SCRIPT_DIR / "segmentation_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # FUNCTION OF REMOVE BACKGROUND
    i = 0
    for image in images:
        parts = split_image(image) 
        masks = []
        for part in parts:
            original_image, pred_mask, foreground, grad_norm = remove_background_morphological_gradient(part)
            masks.append(pred_mask.astype(bool))
        
        #combinar resultados de todos los componentes ---
        if len(masks) == 1:
            mask_final = masks[0].astype(np.uint8)
        else:
            mask_final = np.concatenate(masks, axis=1).astype(bool)
        
            
        mask_uint8 = (mask_final * 255).astype("uint8")
        cv2.imwrite(str(out_dir / f"{i:05d}.png"), mask_uint8)
        i = i + 1

        # Visualización de depuración
        #show_split_debug(image, parts, masks, mask_final)

        predictions.append(mask_final)

    #EVALUATE THE SEGMENTATION WITH THE GROUND TRUTH

    mean_prec, mean_rec, mean_f1 = evaluation(predictions, ground_truths)

    print("Mean Precision:", mean_prec)
    print("Mean Recall:", mean_rec)
    print("Mean F1 Score:", mean_f1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir',
        type=Path,
        default=SCRIPT_DIR.parent / "qsd2_w3",
        help='Path to the dataset directory.'
    )
    data_dir = parser.parse_args().data_dir

    # Check directory
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory.")

    main(data_dir)