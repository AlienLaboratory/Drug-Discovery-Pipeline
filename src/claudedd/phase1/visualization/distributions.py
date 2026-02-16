"""Statistical distribution plots for molecular properties."""

import math
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from claudedd.core.logging import get_logger
from claudedd.core.models import MoleculeDataset
from claudedd.phase1.visualization.utils import (
    create_figure,
    create_subplot_figure,
    save_figure,
)

logger = get_logger("viz.distributions")


def plot_property_histogram(
    dataset: MoleculeDataset,
    property_name: str,
    output_path: str,
    bins: int = 50,
    color: str = "#2196F3",
    title: Optional[str] = None,
    reference_lines: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> str:
    """Plot histogram of a single property across the dataset."""
    values = [
        r.properties[property_name]
        for r in dataset.valid_records
        if property_name in r.properties
        and r.properties[property_name] is not None
        and not (isinstance(r.properties[property_name], float)
                 and math.isnan(r.properties[property_name]))
    ]

    if not values:
        logger.warning(f"No values found for property '{property_name}'")
        return output_path

    fig, ax = create_figure(figsize=figsize)
    ax.hist(values, bins=bins, color=color, alpha=0.7, edgecolor="white")
    ax.set_xlabel(property_name)
    ax.set_ylabel("Count")
    ax.set_title(title or f"Distribution of {property_name}")

    if reference_lines:
        for label, value in reference_lines.items():
            ax.axvline(x=value, color="red", linestyle="--", alpha=0.8, label=label)
        ax.legend()

    return save_figure(fig, output_path)


def plot_property_distributions(
    dataset: MoleculeDataset,
    property_names: List[str],
    output_path: str,
    plot_type: str = "histogram",
    ncols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
) -> str:
    """Plot distributions of multiple properties in a grid layout."""
    n = len(property_names)
    nrows = math.ceil(n / ncols)
    fig, axes = create_subplot_figure(nrows, ncols, figsize=figsize)

    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, prop_name in enumerate(property_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col] if nrows > 1 else axes[0, col]

        values = [
            r.properties[prop_name]
            for r in dataset.valid_records
            if prop_name in r.properties
            and r.properties[prop_name] is not None
            and not (isinstance(r.properties[prop_name], float)
                     and math.isnan(r.properties[prop_name]))
        ]

        if not values:
            ax.set_title(f"{prop_name} (no data)")
            continue

        if plot_type == "histogram":
            ax.hist(values, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
        elif plot_type == "kde":
            sns.kdeplot(values, ax=ax, fill=True)
        elif plot_type == "box":
            ax.boxplot(values, vert=True)
        elif plot_type == "violin":
            ax.violinplot(values, showmeans=True, showmedians=True)

        ax.set_title(prop_name)
        ax.set_xlabel("")

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col] if nrows > 1 else axes[0, col]
        ax.set_visible(False)

    fig.suptitle("Property Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_figure(fig, output_path)


def plot_property_scatter(
    dataset: MoleculeDataset,
    x_property: str,
    y_property: str,
    output_path: str,
    color_property: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> str:
    """Scatter plot of two properties, optionally colored by a third."""
    records = [
        r for r in dataset.valid_records
        if x_property in r.properties and y_property in r.properties
    ]

    x_vals = [r.properties[x_property] for r in records]
    y_vals = [r.properties[y_property] for r in records]

    fig, ax = create_figure(figsize=figsize)

    if color_property and all(color_property in r.properties for r in records):
        c_vals = [r.properties[color_property] for r in records]
        scatter = ax.scatter(x_vals, y_vals, c=c_vals, cmap="viridis",
                             alpha=0.6, s=20, edgecolors="none")
        fig.colorbar(scatter, ax=ax, label=color_property)
    else:
        ax.scatter(x_vals, y_vals, color="#2196F3", alpha=0.6, s=20,
                   edgecolors="none")

    ax.set_xlabel(x_property)
    ax.set_ylabel(y_property)
    ax.set_title(title or f"{x_property} vs {y_property}")

    return save_figure(fig, output_path)


def plot_property_correlation_matrix(
    dataset: MoleculeDataset,
    property_names: Optional[List[str]] = None,
    output_path: str = "",
    figsize: Tuple[int, int] = (12, 10),
) -> str:
    """Correlation heatmap of molecular properties."""
    df = dataset.to_dataframe()

    if property_names:
        available = [p for p in property_names if p in df.columns]
    else:
        available = [c for c in df.columns if df[c].dtype in ("float64", "int64", "float32")]

    if len(available) < 2:
        logger.warning("Not enough numeric properties for correlation matrix")
        return output_path

    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Property Correlation Matrix")

    return save_figure(fig, output_path)
