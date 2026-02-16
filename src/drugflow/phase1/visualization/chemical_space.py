"""Chemical space visualization using dimensionality reduction.

Projects high-dimensional fingerprints to 2D for visualization.
"""

from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from drugflow.core.logging import get_logger
from drugflow.core.models import MoleculeDataset
from drugflow.phase1.visualization.utils import create_figure, save_figure

logger = get_logger("viz.chemical_space")


def _extract_fp_matrix(
    dataset: MoleculeDataset,
    fp_type: str,
) -> Tuple[np.ndarray, List]:
    """Extract fingerprint matrix from dataset. Returns (matrix, records)."""
    valid = [r for r in dataset.valid_records if fp_type in r.fingerprints]
    if not valid:
        raise ValueError(f"No fingerprints of type '{fp_type}' found in dataset")

    matrix = np.array([r.fingerprints[fp_type] for r in valid], dtype=np.float32)
    return matrix, valid


def plot_chemical_space_pca(
    dataset: MoleculeDataset,
    fp_type: str = "morgan_r2_2048",
    output_path: str = "",
    color_property: Optional[str] = None,
    title: str = "Chemical Space (PCA)",
    figsize: Tuple[int, int] = (10, 8),
    explained_variance: bool = True,
) -> str:
    """PCA projection of fingerprints to 2D."""
    matrix, valid = _extract_fp_matrix(dataset, fp_type)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)

    fig, ax = create_figure(figsize=figsize)

    if color_property:
        c_vals = []
        mask = []
        for i, rec in enumerate(valid):
            val = rec.properties.get(color_property)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                c_vals.append(val)
                mask.append(True)
            else:
                c_vals.append(0)
                mask.append(False)

        if any(mask):
            c_arr = np.array(c_vals)
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=c_arr, cmap="viridis", alpha=0.6, s=15, edgecolors="none",
            )
            fig.colorbar(scatter, ax=ax, label=color_property)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=15,
                       color="#2196F3", edgecolors="none")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=15,
                   color="#2196F3", edgecolors="none")

    xlabel = f"PC1"
    ylabel = f"PC2"
    if explained_variance:
        xlabel += f" ({pca.explained_variance_ratio_[0]:.1%})"
        ylabel += f" ({pca.explained_variance_ratio_[1]:.1%})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return save_figure(fig, output_path)


def plot_chemical_space_tsne(
    dataset: MoleculeDataset,
    fp_type: str = "morgan_r2_2048",
    output_path: str = "",
    color_property: Optional[str] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    title: str = "Chemical Space (t-SNE)",
    figsize: Tuple[int, int] = (10, 8),
) -> str:
    """t-SNE projection of fingerprints to 2D."""
    matrix, valid = _extract_fp_matrix(dataset, fp_type)

    # Adjust perplexity if dataset is small
    actual_perplexity = min(perplexity, max(5.0, len(valid) / 4))

    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        n_iter=n_iter,
        random_state=random_state,
    )
    coords = tsne.fit_transform(matrix)

    fig, ax = create_figure(figsize=figsize)

    if color_property:
        c_vals = [
            rec.properties.get(color_property, 0) for rec in valid
        ]
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=c_vals, cmap="viridis", alpha=0.6, s=15, edgecolors="none",
        )
        fig.colorbar(scatter, ax=ax, label=color_property)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=15,
                   color="#2196F3", edgecolors="none")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)

    return save_figure(fig, output_path)


def plot_multiple_datasets(
    datasets: List[MoleculeDataset],
    labels: List[str],
    fp_type: str = "morgan_r2_2048",
    method: str = "pca",
    output_path: str = "",
    figsize: Tuple[int, int] = (12, 8),
) -> str:
    """Overlay multiple datasets in the same chemical space plot."""
    all_fps = []
    dataset_indices = []

    for i, ds in enumerate(datasets):
        for rec in ds.valid_records:
            if fp_type in rec.fingerprints:
                all_fps.append(rec.fingerprints[fp_type])
                dataset_indices.append(i)

    if not all_fps:
        raise ValueError("No fingerprints found in any dataset")

    matrix = np.array(all_fps, dtype=np.float32)
    dataset_indices = np.array(dataset_indices)

    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        perp = min(30.0, max(5.0, len(matrix) / 4))
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42)

    coords = reducer.fit_transform(matrix)

    fig, ax = create_figure(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    for i, label in enumerate(labels):
        mask = dataset_indices == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            alpha=0.6, s=15, label=label, color=colors[i],
            edgecolors="none",
        )

    ax.legend()
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"Chemical Space Comparison ({method.upper()})")

    return save_figure(fig, output_path)
