"""Shared plot styling and save helpers."""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_plot_style():
    """Configure matplotlib defaults for publication-quality plots."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def save_figure(
    fig: plt.Figure,
    path: str,
    dpi: int = 300,
    tight: bool = True,
) -> str:
    """Save figure and return path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def create_figure(
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a styled figure with single axes."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def create_subplot_figure(
    nrows: int,
    ncols: int,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, any]:
    """Create a styled figure with multiple subplots."""
    setup_plot_style()
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes
