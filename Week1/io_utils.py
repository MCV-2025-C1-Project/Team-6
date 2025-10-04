import pickle
from typing import Any, List
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Read data from pickle file
def read_pickle(file_path: Path) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


# Write data to a pickle file
def write_pickle(data: Any, file_path: Path) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def read_images(dir_path: Path) -> List[np.ndarray]:
    """
    Reads all JPG images from the given directory using OpenCV (sorted by their filenames).

    Args:
        dir_path (Path):    Path to the image directory.

    Returns:
        List[np.ndarray]:   A list containing the images in RGB.
    """
    return [
        cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        for img_path in sorted(dir_path.glob("*.jpg"))
    ]


def Plot_comparison_descriptors_simils(results: np.ndarray,
                                       title: str = "Comparación: Descriptores vs Métricas",
                                       annotate: bool = True):
    """
    Dibuja y MUESTRA un heatmap para una matriz (2, 7):
      filas = [RGB, HSV]
      columnas = [euclidean, l1, chi2, hist_intersection, hellinger, cosine, bhattacharyya]
    """
    # Validaciones
    if not isinstance(results, np.ndarray):
        raise TypeError("results debe ser un numpy.ndarray")
    if results.shape != (2, 7):
        raise ValueError(f"Shape inválido {results.shape}. Debe ser (2, 7).")

    descriptor_names = ["RGB", "HSV"]
    similarity_names = [
        "Euclidean", "L1", "Chi-2", "Hist. Intersection",
        "Hellinger", "Cosine", "Bhattacharyya"
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(results, cmap="viridis", aspect="auto")

    ax.set_xticks(np.arange(len(similarity_names)))
    ax.set_yticks(np.arange(len(descriptor_names)))
    ax.set_xticklabels(similarity_names)
    ax.set_yticklabels(descriptor_names)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    if annotate:
        vmin, vmax = np.nanmin(results), np.nanmax(results)
        mid = (vmin + vmax) / 2.0
        for i in range(2):
            for j in range(7):
                val = results[i, j]
                ax.text(j, i, f"{val:.3f}",
                        ha="center", va="center",
                        color=("white" if val < mid else "black"))

    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ejemplo de uso
    # (opcional) ejemplo más "realista" con HSV > RGB en intersección/chi2
    results = np.array([
        [0.31, 0.37, 0.33, 0.35, 0.32, 0.30, 0.31],  # RGB
        [0.33, 0.47, 0.46, 0.47, 0.44, 0.33, 0.47],  # HSV
    ], dtype=float)

    Plot_comparison_descriptors_simils(
        results,
        title="MAP@K: RGB vs HSV con 7 métricas",
        annotate=True
    )