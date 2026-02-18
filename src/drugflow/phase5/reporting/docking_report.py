"""Docking results visualization and analysis report.

Generates publication-quality figures, data tables, and statistical
summaries for molecular docking results. Produces 8 PNG plots, a
ranked CSV, an enhanced JSON report, and a text summary.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from drugflow.phase1.visualization.utils import (
    create_figure,
    create_subplot_figure,
    save_figure,
    setup_plot_style,
)

logger = logging.getLogger(__name__)

# ── DrugFlow Color Palette ──────────────────────────────────────────

COLORS = {
    "blue": "#2196F3",
    "green": "#4CAF50",
    "orange": "#FF9800",
    "red": "#F44336",
    "light_blue": "#90CAF9",
    "purple": "#9C27B0",
    "dark_blue": "#1565C0",
    "teal": "#009688",
    "dark_gray": "#37474F",
}

ADMET_CLASS_COLORS = {
    "favorable": "#4CAF50",
    "moderate": "#FF9800",
    "poor": "#F44336",
}

RISK_BG_COLORS = {
    "green": "#C8E6C9",
    "yellow": "#FFF9C4",
    "red": "#FFCDD2",
}

RISK_TEXT_COLORS = {
    "green": "#2E7D32",
    "yellow": "#F57F17",
    "red": "#C62828",
}

RADAR_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

# ── Statistics ──────────────────────────────────────────────────────


def _compute_docking_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive statistics from docking results."""
    stats: Dict[str, Any] = {}

    # Vina scores
    if "vina_score" in df.columns:
        vs = df["vina_score"].dropna()
        stats["vina"] = {
            "n": int(len(vs)),
            "mean": round(float(vs.mean()), 3),
            "std": round(float(vs.std()), 3),
            "min": round(float(vs.min()), 3),
            "max": round(float(vs.max()), 3),
            "median": round(float(vs.median()), 3),
            "q1": round(float(vs.quantile(0.25)), 3),
            "q3": round(float(vs.quantile(0.75)), 3),
            "iqr": round(float(vs.quantile(0.75) - vs.quantile(0.25)), 3),
        }
        # Binding tiers
        stats["vina"]["n_strong"] = int((vs <= -3.5).sum())
        stats["vina"]["n_moderate"] = int(((vs > -3.5) & (vs <= -3.0)).sum())
        stats["vina"]["n_weak"] = int((vs > -3.0).sum())
    else:
        stats["vina"] = {"n": 0}

    # Potency
    pic50_col = None
    for col in ["predicted_pIC50", "pIC50", "predicted_activity"]:
        if col in df.columns:
            pic50_col = col
            break
    if pic50_col:
        ps = df[pic50_col].dropna()
        stats["potency"] = {
            "column": pic50_col,
            "mean": round(float(ps.mean()), 3),
            "std": round(float(ps.std()), 3),
            "min": round(float(ps.min()), 3),
            "max": round(float(ps.max()), 3),
            "median": round(float(ps.median()), 3),
        }
    if "predicted_IC50_nM" in df.columns:
        ic = df["predicted_IC50_nM"].dropna()
        stats.setdefault("potency", {})
        stats["potency"]["ic50_nm_min"] = round(float(ic.min()), 1)
        stats["potency"]["ic50_nm_max"] = round(float(ic.max()), 1)

    # ADMET
    if "admet_score" in df.columns:
        ads = df["admet_score"].dropna()
        stats["admet"] = {
            "mean": round(float(ads.mean()), 3),
            "std": round(float(ads.std()), 3),
            "min": round(float(ads.min()), 3),
            "max": round(float(ads.max()), 3),
        }
    if "admet_class" in df.columns:
        vc = df["admet_class"].value_counts()
        stats.setdefault("admet", {})
        stats["admet"]["favorable"] = int(vc.get("favorable", 0))
        stats["admet"]["moderate"] = int(vc.get("moderate", 0))
        stats["admet"]["poor"] = int(vc.get("poor", 0))

    # Risk flags
    risk_cols = {
        "herg": "admet_herg_risk",
        "ames": "admet_ames_risk",
        "hepatotox": "admet_hepatotox_risk",
        "cyp": "admet_cyp_inhibition_risk",
    }
    risk_summary = {}
    for name, col in risk_cols.items():
        if col in df.columns:
            vc = df[col].value_counts()
            risk_summary[name] = {
                "green": int(vc.get("green", 0)),
                "yellow": int(vc.get("yellow", 0)),
                "red": int(vc.get("red", 0)),
            }
            # Handle non-standard values
            for val in ["low", "moderate", "high"]:
                if val in vc.index:
                    mapping = {"low": "green", "moderate": "yellow", "high": "red"}
                    risk_summary[name][mapping[val]] = risk_summary[name].get(
                        mapping[val], 0
                    ) + int(vc[val])
    stats["risk_flags"] = risk_summary

    # Molecular properties
    prop_stats = {}
    for col in ["MolWt", "LogP", "QED", "RingCount"]:
        if col in df.columns:
            s = df[col].dropna()
            prop_stats[col] = {
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
            }
    stats["properties"] = prop_stats

    # Correlations
    correlations = {}
    try:
        from scipy.stats import spearmanr

        if "vina_score" in df.columns and pic50_col and len(df) > 3:
            mask = df[["vina_score", pic50_col]].dropna()
            if len(mask) > 3:
                rho, p = spearmanr(mask["vina_score"], mask[pic50_col])
                correlations["vina_vs_pic50"] = {
                    "spearman_rho": round(float(rho), 3),
                    "p_value": round(float(p), 4),
                }
        if "vina_score" in df.columns and "admet_score" in df.columns:
            mask = df[["vina_score", "admet_score"]].dropna()
            if len(mask) > 3:
                rho, p = spearmanr(mask["vina_score"], mask["admet_score"])
                correlations["vina_vs_admet"] = {
                    "spearman_rho": round(float(rho), 3),
                    "p_value": round(float(p), 4),
                }
    except ImportError:
        pass
    stats["correlations"] = correlations

    return stats


def _compute_population_statistics(admet_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute ADMET population statistics for the full candidate pool."""
    stats: Dict[str, Any] = {"total": len(admet_df)}
    if "admet_class" in admet_df.columns:
        vc = admet_df["admet_class"].value_counts()
        stats["favorable"] = int(vc.get("favorable", 0))
        stats["moderate"] = int(vc.get("moderate", 0))
        stats["poor"] = int(vc.get("poor", 0))
    if "admet_score" in admet_df.columns:
        s = admet_df["admet_score"].dropna()
        stats["mean_score"] = round(float(s.mean()), 3)
        stats["score_range"] = [round(float(s.min()), 3), round(float(s.max()), 3)]

    # Domain sub-scores
    for domain in ["absorption", "distribution", "metabolism", "excretion", "toxicity"]:
        col = f"admet_{domain}_score"
        if col in admet_df.columns:
            s = admet_df[col].dropna()
            stats[f"{domain}_mean"] = round(float(s.mean()), 3)

    return stats


# ── Plot Functions ──────────────────────────────────────────────────


def _plot_ranked_candidates_table(
    df: pd.DataFrame, plots_dir: str, top_n: int = 15
) -> str:
    """Plot 1: Publication-quality ranked candidates table."""
    setup_plot_style()
    top = df.head(top_n).copy()
    n = len(top)

    # Columns to display
    display_cols = []
    col_labels = []
    col_fmts = []

    for col, label, fmt in [
        ("rank", "Rank", "{:.0f}"),
        ("record_id", "ID", "{}"),
        ("vina_score", "Vina\n(kcal/mol)", "{:.2f}"),
        ("predicted_pIC50", "pIC50", "{:.2f}"),
        ("predicted_IC50_nM", "IC50\n(nM)", "{:.1f}"),
        ("admet_score", "ADMET\nScore", "{:.3f}"),
        ("admet_class", "ADMET\nClass", "{}"),
        ("MolWt", "MW", "{:.0f}"),
        ("LogP", "LogP", "{:.1f}"),
        ("QED", "QED", "{:.2f}"),
        ("admet_herg_risk", "hERG", "{}"),
        ("admet_ames_risk", "AMES", "{}"),
        ("admet_hepatotox_risk", "Hepato-\ntox", "{}"),
    ]:
        if col in top.columns:
            display_cols.append(col)
            col_labels.append(label)
            col_fmts.append(fmt)

    # Build cell text
    cell_text = []
    for _, row in top.iterrows():
        cells = []
        for col, fmt in zip(display_cols, col_fmts):
            val = row.get(col, "")
            try:
                if pd.notna(val) and val != "":
                    cells.append(fmt.format(val))
                else:
                    cells.append("—")
            except (ValueError, TypeError):
                cells.append(str(val)[:12])
        cell_text.append(cells)

    fig_height = max(4, 0.55 * n + 2.5)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(COLORS["dark_gray"])
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Style data rows
    admet_class_idx = display_cols.index("admet_class") if "admet_class" in display_cols else -1
    risk_indices = []
    for rc in ["admet_herg_risk", "admet_ames_risk", "admet_hepatotox_risk"]:
        if rc in display_cols:
            risk_indices.append(display_cols.index(rc))

    for i in range(n):
        row_color = "#FFFFFF" if i % 2 == 0 else "#F5F5F5"
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor(row_color)
            cell.set_edgecolor("#E0E0E0")

            # Color ADMET class cells
            if j == admet_class_idx:
                cls = cell_text[i][j]
                if cls in ADMET_CLASS_COLORS:
                    bg = {"favorable": "#E8F5E9", "moderate": "#FFF8E1", "poor": "#FFEBEE"}
                    cell.set_facecolor(bg.get(cls, row_color))

            # Color risk flag text
            if j in risk_indices:
                risk_val = cell_text[i][j].lower()
                if risk_val in RISK_TEXT_COLORS:
                    cell.set_text_props(color=RISK_TEXT_COLORS[risk_val], fontweight="bold")

    ax.set_title(
        f"Top {top_n} Docking Candidates",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.text(
        0.5, -0.02,
        "Ranked by AutoDock Vina binding energy (lower = stronger binding)",
        ha="center", va="top", transform=ax.transAxes, fontsize=9, color="#666666",
    )

    path = os.path.join(plots_dir, "01_ranked_candidates_table.png")
    return save_figure(fig, path)


def _plot_vina_score_distribution(df: pd.DataFrame, plots_dir: str) -> str:
    """Plot 2: Vina binding energy distribution histogram."""
    if "vina_score" not in df.columns:
        return ""

    scores = df["vina_score"].dropna()
    fig, ax = create_figure(figsize=(10, 6))

    ax.hist(scores, bins=15, color=COLORS["blue"], alpha=0.8, edgecolor="white", linewidth=0.8)

    # Reference lines
    mean_val = scores.mean()
    median_val = scores.median()
    ax.axvline(mean_val, color=COLORS["orange"], ls="--", lw=2, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color=COLORS["red"], ls="--", lw=2, label=f"Median: {median_val:.2f}")

    # Stats box
    stats_text = (
        f"n = {len(scores)}\n"
        f"Mean: {mean_val:.3f} ± {scores.std():.3f}\n"
        f"Range: [{scores.min():.2f}, {scores.max():.2f}]"
    )
    ax.text(
        0.02, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", alpha=0.9),
    )

    ax.set_xlabel("Vina Binding Energy (kcal/mol)", fontsize=12)
    ax.set_ylabel("Number of Candidates", fontsize=12)
    ax.set_title("AutoDock Vina Binding Energy Distribution", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(plots_dir, "02_vina_score_distribution.png")
    return save_figure(fig, path)


def _plot_admet_radar_top_n(
    df: pd.DataFrame,
    admet_df: Optional[pd.DataFrame],
    plots_dir: str,
    top_n: int = 5,
) -> str:
    """Plot 3: ADMET radar/spider chart for top N candidates."""
    setup_plot_style()
    categories = ["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"]
    score_cols = [f"admet_{c.lower()}_score" for c in categories]

    top = df.head(top_n).copy()

    # Try to get domain sub-scores from admet_all CSV
    if admet_df is not None and all(c in admet_df.columns for c in score_cols):
        if "smiles" in top.columns and "smiles" in admet_df.columns:
            merged = top.merge(
                admet_df[["smiles"] + score_cols].drop_duplicates("smiles"),
                on="smiles", how="left", suffixes=("", "_full"),
            )
            for sc in score_cols:
                full_col = sc + "_full"
                if full_col in merged.columns:
                    merged[sc] = merged[full_col].fillna(merged.get(sc, 0.5))
            top = merged

    # Fallback: derive approximate sub-scores from risk flags
    has_scores = all(c in top.columns for c in score_cols)
    if not has_scores:
        risk_map = {"green": 1.0, "yellow": 0.5, "red": 0.0}
        for _, row in top.iterrows():
            if "admet_absorption_score" not in top.columns:
                pass  # will be handled below

        # Simple derivation
        def _risk_val(v):
            if isinstance(v, str):
                return risk_map.get(v.lower(), 0.5)
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            return 0.5

        for sc, risk_cols_map in [
            ("admet_absorption_score", ["admet_caco2_class", "admet_hia_class"]),
            ("admet_distribution_score", ["admet_bbb_penetrant"]),
            ("admet_metabolism_score", ["admet_cyp_inhibition_risk", "admet_metabolic_stability"]),
            ("admet_excretion_score", []),
            ("admet_toxicity_score", ["admet_herg_risk", "admet_ames_risk", "admet_hepatotox_risk"]),
        ]:
            if sc not in top.columns:
                avail = [c for c in risk_cols_map if c in top.columns]
                if avail:
                    top[sc] = top[avail].map(
                        lambda v: _risk_val(v) if isinstance(v, str) else (
                            {"high": 1.0, "moderate": 0.5, "low": 0.0}.get(str(v).lower(), 0.5)
                            if isinstance(v, str) else 0.5
                        )
                    ).mean(axis=1)
                else:
                    top[sc] = 0.5

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})

    for idx in range(min(top_n, len(top))):
        row = top.iloc[idx]
        values = [float(row.get(sc, 0.5)) for sc in score_cols]
        values += values[:1]

        label = str(row.get("record_id", f"#{idx+1}"))
        vina_str = f" ({row.get('vina_score', 0):.2f})" if "vina_score" in row.index else ""
        color = RADAR_COLORS[idx % len(RADAR_COLORS)]

        ax.plot(angles, values, "o-", linewidth=2, color=color, label=f"{label}{vina_str}")
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"], fontsize=8, color="#666")
    ax.set_title("ADMET Domain Profiles — Top 5 Candidates", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9, title="Candidate (Vina)")

    path = os.path.join(plots_dir, "03_admet_radar_top5.png")
    return save_figure(fig, path)


def _plot_score_correlations(df: pd.DataFrame, plots_dir: str) -> str:
    """Plot 4: Three-panel correlation scatter plots."""
    fig, axes = create_subplot_figure(1, 3, figsize=(18, 6))

    # Color mapping
    color_map = ADMET_CLASS_COLORS
    default_color = COLORS["blue"]

    def _get_colors(series):
        return [color_map.get(v, default_color) for v in series]

    panels = []
    if "vina_score" in df.columns and "predicted_pIC50" in df.columns:
        panels.append(("vina_score", "predicted_pIC50", "Vina Score (kcal/mol)", "Predicted pIC50"))
    if "vina_score" in df.columns and "admet_score" in df.columns:
        panels.append(("vina_score", "admet_score", "Vina Score (kcal/mol)", "ADMET Score"))
    if "predicted_pIC50" in df.columns and "admet_score" in df.columns:
        panels.append(("predicted_pIC50", "admet_score", "Predicted pIC50", "ADMET Score"))

    for i, ax in enumerate(axes.flat):
        if i < len(panels):
            xcol, ycol, xlabel, ylabel = panels[i]
            mask = df[[xcol, ycol]].dropna()
            colors = _get_colors(df.loc[mask.index, "admet_class"]) if "admet_class" in df.columns else default_color

            # Size based on vina for panel 3
            sizes = 80
            if i == 2 and "vina_score" in df.columns:
                vs = df.loc[mask.index, "vina_score"].abs()
                sizes = 20 + 180 * (vs - vs.min()) / max(vs.max() - vs.min(), 0.001)

            ax.scatter(mask[xcol], mask[ycol], c=colors, s=sizes, alpha=0.8,
                       edgecolors="white", linewidth=0.5, zorder=5)

            # Trendline
            if len(mask) > 2:
                z = np.polyfit(mask[xcol], mask[ycol], 1)
                p = np.poly1d(z)
                x_line = np.linspace(mask[xcol].min(), mask[xcol].max(), 50)
                ax.plot(x_line, p(x_line), "--", color="#999999", lw=1.5, zorder=3)

                # Spearman
                try:
                    from scipy.stats import spearmanr
                    rho, pval = spearmanr(mask[xcol], mask[ycol])
                    ax.text(
                        0.05, 0.95, f"ρ = {rho:.3f}\np = {pval:.3f}",
                        transform=ax.transAxes, fontsize=9, va="top",
                        bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8),
                    )
                except ImportError:
                    pass

            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")

    # Legend
    if "admet_class" in df.columns:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=ADMET_CLASS_COLORS[c], label=c.capitalize())
            for c in ["favorable", "moderate", "poor"]
            if c in df["admet_class"].values
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Score Correlations", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    path = os.path.join(plots_dir, "04_score_correlations.png")
    return save_figure(fig, path)


def _plot_admet_class_distribution(
    docking_df: pd.DataFrame,
    admet_df: Optional[pd.DataFrame],
    plots_dir: str,
) -> str:
    """Plot 5: ADMET class donut charts."""
    setup_plot_style()
    has_all = admet_df is not None and "admet_class" in admet_df.columns
    ncols = 2 if has_all else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    def _donut(ax, data, title, center_text):
        class_order = ["favorable", "moderate", "poor"]
        counts = []
        colors = []
        labels = []
        total = len(data)
        vc = data["admet_class"].value_counts()
        for cls in class_order:
            c = int(vc.get(cls, 0))
            if c > 0:
                counts.append(c)
                colors.append(ADMET_CLASS_COLORS[cls])
                pct = 100 * c / total if total > 0 else 0
                labels.append(f"{cls.capitalize()}\n{c} ({pct:.1f}%)")

        wedges, texts = ax.pie(
            counts, labels=labels, colors=colors,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
            startangle=90, textprops={"fontsize": 10},
        )
        ax.text(0, 0, center_text, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    if has_all:
        _donut(axes[0], admet_df, f"All Novel Candidates (n={len(admet_df)})",
               f"{len(admet_df)}\nCandidates")

    dock_idx = 1 if has_all else 0
    if "admet_class" in docking_df.columns:
        _donut(axes[dock_idx], docking_df,
               f"Docked Candidates (n={len(docking_df)})",
               f"{len(docking_df)}\nDocked")

    fig.suptitle("ADMET Classification Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(plots_dir, "05_admet_class_distribution.png")
    return save_figure(fig, path)


def _plot_risk_flag_heatmap(df: pd.DataFrame, plots_dir: str) -> str:
    """Plot 6: Safety risk flag heatmap for all docked candidates."""
    setup_plot_style()

    risk_cols_map = {
        "admet_herg_risk": "hERG",
        "admet_ames_risk": "AMES",
        "admet_hepatotox_risk": "Hepatotox",
        "admet_cyp_inhibition_risk": "CYP Inhib",
        "admet_bbb_penetrant": "BBB",
        "admet_metabolic_stability": "Metabolic\nStab",
    }

    avail_cols = [c for c in risk_cols_map if c in df.columns]
    if not avail_cols:
        return ""

    labels = [risk_cols_map[c] for c in avail_cols]

    # Map values to numeric
    def _to_numeric(val):
        if isinstance(val, bool):
            return 2.0 if val else 0.0
        s = str(val).strip().lower()
        mapping = {"green": 2, "yellow": 1, "red": 0, "high": 2, "moderate": 1, "low": 0,
                   "true": 2, "false": 0}
        return mapping.get(s, 1.0)

    numeric_data = df[avail_cols].map(_to_numeric).values
    annot_data = df[avail_cols].map(lambda v: str(v)[:8]).values

    n_rows = len(df)
    fig_height = max(6, 0.4 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(max(10, 2 * len(avail_cols)), fig_height))

    cmap = ListedColormap(["#FFCDD2", "#FFF9C4", "#C8E6C9"])
    im = ax.imshow(numeric_data, cmap=cmap, aspect="auto", vmin=0, vmax=2)

    # Annotations
    for i in range(n_rows):
        for j in range(len(avail_cols)):
            text = annot_data[i, j]
            color = "#333333"
            if text.lower() in ("red", "false"):
                color = RISK_TEXT_COLORS["red"]
            elif text.lower() in ("yellow", "moderate"):
                color = RISK_TEXT_COLORS["yellow"]
            elif text.lower() in ("green", "high", "true"):
                color = RISK_TEXT_COLORS["green"]
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    y_labels = df["record_id"].tolist() if "record_id" in df.columns else [f"#{i+1}" for i in range(n_rows)]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title("Safety Risk Flag Heatmap — Docked Candidates", fontsize=14, fontweight="bold", pad=15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C8E6C9", edgecolor="#999", label="Green / Favorable"),
        Patch(facecolor="#FFF9C4", edgecolor="#999", label="Yellow / Moderate"),
        Patch(facecolor="#FFCDD2", edgecolor="#999", label="Red / Unfavorable"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=3, fontsize=9)

    fig.tight_layout()
    path = os.path.join(plots_dir, "06_risk_flag_heatmap.png")
    return save_figure(fig, path)


def _plot_integrated_dashboard(
    df: pd.DataFrame,
    admet_df: Optional[pd.DataFrame],
    stats: Dict,
    plots_dir: str,
) -> str:
    """Plot 7: Multi-panel integrated dashboard."""
    setup_plot_style()
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # [0,0] Horizontal bar chart — top 10 Vina scores
    ax1 = fig.add_subplot(gs[0, 0])
    top10 = df.head(10).copy()
    if "vina_score" in top10.columns:
        y_pos = range(len(top10))
        ids = top10["record_id"].tolist() if "record_id" in top10.columns else [f"#{i}" for i in range(len(top10))]
        vs = top10["vina_score"].values
        # Color gradient
        norm_vs = (vs - vs.max()) / (vs.min() - vs.max() + 1e-9)
        bar_colors = [plt.cm.Blues(0.4 + 0.5 * v) for v in norm_vs]
        ax1.barh(y_pos, vs, color=bar_colors, edgecolor="white", height=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(ids, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("Vina Score (kcal/mol)", fontsize=10)
        ax1.set_title("Top 10 Binding Energies", fontsize=12, fontweight="bold")
        if "vina" in stats:
            ax1.axvline(stats["vina"]["mean"], color=COLORS["red"], ls="--", lw=1.5, alpha=0.7)
    ax1.grid(True, alpha=0.3, axis="x")

    # [0,1] Scatter — Vina vs pIC50
    ax2 = fig.add_subplot(gs[0, 1])
    if "vina_score" in df.columns and "predicted_pIC50" in df.columns:
        colors = [ADMET_CLASS_COLORS.get(c, COLORS["blue"]) for c in df.get("admet_class", ["blue"] * len(df))]
        ax2.scatter(df["vina_score"], df["predicted_pIC50"], c=colors, s=60, alpha=0.8,
                    edgecolors="white", linewidth=0.5)
        ax2.set_xlabel("Vina Score (kcal/mol)", fontsize=10)
        ax2.set_ylabel("Predicted pIC50", fontsize=10)
    ax2.set_title("Vina Score vs Potency", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # [0,2] ADMET donut
    ax3 = fig.add_subplot(gs[0, 2])
    if "admet_class" in df.columns:
        vc = df["admet_class"].value_counts()
        class_order = ["favorable", "moderate", "poor"]
        counts = [int(vc.get(c, 0)) for c in class_order if c in vc.index]
        clrs = [ADMET_CLASS_COLORS[c] for c in class_order if c in vc.index]
        lbls = [f"{c.capitalize()}: {int(vc.get(c, 0))}" for c in class_order if c in vc.index]
        ax3.pie(counts, labels=lbls, colors=clrs,
                wedgeprops=dict(width=0.4, edgecolor="white"), startangle=90,
                textprops={"fontsize": 9})
        ax3.text(0, 0, f"{len(df)}", ha="center", va="center", fontsize=16, fontweight="bold")
    ax3.set_title("ADMET Classes (Docked)", fontsize=12, fontweight="bold")

    # [1,0] Property box plots
    ax4 = fig.add_subplot(gs[1, 0])
    prop_data = []
    prop_labels = []
    prop_colors = []
    for col, lbl, clr in [("MolWt", "MW", COLORS["blue"]), ("LogP", "LogP", COLORS["green"]), ("QED", "QED", COLORS["orange"])]:
        if col in df.columns:
            prop_data.append(df[col].dropna().values)
            prop_labels.append(lbl)
            prop_colors.append(clr)
    if prop_data:
        # Normalize for display
        bp = ax4.boxplot(prop_data, tick_labels=prop_labels, patch_artist=True,
                         widths=0.5, showfliers=True)
        for patch, color in zip(bp["boxes"], prop_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        # Jitter
        for i, data in enumerate(prop_data):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax4.scatter(x, data, s=15, alpha=0.5, color=prop_colors[i], zorder=5)
    ax4.set_title("Molecular Properties", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    # [1,1] Mini risk heatmap (top 10)
    ax5 = fig.add_subplot(gs[1, 1])
    risk_cols = ["admet_herg_risk", "admet_ames_risk", "admet_hepatotox_risk", "admet_cyp_inhibition_risk"]
    avail_risk = [c for c in risk_cols if c in df.columns]
    if avail_risk:
        top10_risk = df.head(10)[avail_risk].copy()

        def _rmap(v):
            s = str(v).strip().lower()
            return {"green": 2, "yellow": 1, "red": 0, "high": 2, "moderate": 1, "low": 0}.get(s, 1)

        numeric = top10_risk.map(_rmap).values
        cmap = ListedColormap(["#FFCDD2", "#FFF9C4", "#C8E6C9"])
        ax5.imshow(numeric, cmap=cmap, aspect="auto", vmin=0, vmax=2)
        short_labels = {"admet_herg_risk": "hERG", "admet_ames_risk": "AMES",
                        "admet_hepatotox_risk": "Hepato", "admet_cyp_inhibition_risk": "CYP"}
        ax5.set_xticks(range(len(avail_risk)))
        ax5.set_xticklabels([short_labels.get(c, c) for c in avail_risk], fontsize=9)
        ids = df.head(10)["record_id"].tolist() if "record_id" in df.columns else list(range(10))
        ax5.set_yticks(range(len(top10_risk)))
        ax5.set_yticklabels(ids, fontsize=7)
        # Annotations
        for i in range(len(top10_risk)):
            for j in range(len(avail_risk)):
                txt = str(top10_risk.iloc[i, j])[:6]
                ax5.text(j, i, txt, ha="center", va="center", fontsize=7)
    ax5.set_title("Safety Flags (Top 10)", fontsize=12, fontweight="bold")

    # [1,2] Key stats text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    lines = ["KEY STATISTICS", "─" * 30]
    if "vina" in stats and stats["vina"]["n"] > 0:
        v = stats["vina"]
        lines.append(f"Candidates docked:  {v['n']}")
        lines.append(f"Vina score range:   {v['min']:.2f} to {v['max']:.2f}")
        lines.append(f"Mean ± std:         {v['mean']:.3f} ± {v['std']:.3f}")
        lines.append(f"Strong (≤-3.5):     {v.get('n_strong', 0)}")
        lines.append(f"Moderate (-3.0—3.5): {v.get('n_moderate', 0)}")
    if "potency" in stats:
        p = stats["potency"]
        lines.append("")
        lines.append(f"pIC50 range:        {p.get('min', 0):.2f} – {p.get('max', 0):.2f}")
        lines.append(f"Mean pIC50:         {p.get('mean', 0):.2f}")
    if "admet" in stats:
        a = stats["admet"]
        lines.append("")
        lines.append(f"ADMET favorable:    {a.get('favorable', 0)}")
        lines.append(f"ADMET moderate:     {a.get('moderate', 0)}")
        lines.append(f"Mean ADMET score:   {a.get('mean', 0):.3f}")
    if "correlations" in stats:
        c = stats["correlations"]
        if "vina_vs_pic50" in c:
            lines.append("")
            lines.append(f"Vina–pIC50 ρ:       {c['vina_vs_pic50']['spearman_rho']:.3f}")

    text = "\n".join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=10,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA", edgecolor="#E0E0E0"))

    fig.suptitle("Docking Study — Integrated Results Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)

    path = os.path.join(plots_dir, "07_integrated_dashboard.png")
    return save_figure(fig, path)


def _plot_summary_statistics_table(stats: Dict, plots_dir: str) -> str:
    """Plot 8: Summary statistics rendered as a publication-quality table image."""
    setup_plot_style()

    rows = []
    # Section: Docking Results
    rows.append(("DOCKING RESULTS", "", ""))
    if "vina" in stats and stats["vina"]["n"] > 0:
        v = stats["vina"]
        rows.append(("  Candidates docked", str(v["n"]), ""))
        rows.append(("  Vina mean ± std", f"{v['mean']:.3f} ± {v['std']:.3f}", "kcal/mol"))
        rows.append(("  Vina range", f"[{v['min']:.2f}, {v['max']:.2f}]", "kcal/mol"))
        rows.append(("  Vina median", f"{v['median']:.3f}", "kcal/mol"))
        rows.append(("  Strong binding (≤-3.5)", str(v.get("n_strong", 0)), ""))
        rows.append(("  Moderate binding (-3.0 to -3.5)", str(v.get("n_moderate", 0)), ""))

    # Section: Potency
    rows.append(("PREDICTED POTENCY", "", ""))
    if "potency" in stats:
        p = stats["potency"]
        rows.append(("  pIC50 mean ± std", f"{p.get('mean', 0):.3f} ± {p.get('std', 0):.3f}", ""))
        rows.append(("  pIC50 range", f"[{p.get('min', 0):.2f}, {p.get('max', 0):.2f}]", ""))
        if "ic50_nm_min" in p:
            rows.append(("  IC50 range", f"[{p['ic50_nm_min']:.1f}, {p['ic50_nm_max']:.1f}]", "nM"))

    # Section: ADMET
    rows.append(("ADMET PROFILE", "", ""))
    if "admet" in stats:
        a = stats["admet"]
        rows.append(("  Mean score", f"{a.get('mean', 0):.3f}", "0–1 scale"))
        rows.append(("  Favorable", str(a.get("favorable", 0)), ""))
        rows.append(("  Moderate", str(a.get("moderate", 0)), ""))
        rows.append(("  Poor", str(a.get("poor", 0)), ""))

    # Section: Properties
    rows.append(("DRUG-LIKENESS", "", ""))
    for prop_name, unit in [("MolWt", "g/mol"), ("LogP", ""), ("QED", "0–1"), ("RingCount", "")]:
        if prop_name in stats.get("properties", {}):
            p = stats["properties"][prop_name]
            rows.append((f"  {prop_name} mean", f"{p['mean']:.2f} ± {p['std']:.2f}", unit))

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * n_rows + 2)))
    ax.axis("off")

    cell_text = [[r[0], r[1], r[2]] for r in rows]
    col_labels = ["Metric", "Value", "Unit"]

    table = ax.table(
        cellText=cell_text, colLabels=col_labels,
        cellLoc="left", loc="center",
        colWidths=[0.45, 0.35, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Header styling
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor(COLORS["dark_gray"])
        cell.set_text_props(color="white", fontweight="bold")

    # Row styling
    for i in range(n_rows):
        is_section = rows[i][1] == "" and rows[i][2] == "" and not rows[i][0].startswith("  ")
        for j in range(3):
            cell = table[i + 1, j]
            if is_section:
                cell.set_facecolor("#E3F2FD")
                cell.set_text_props(fontweight="bold", fontsize=9)
            else:
                cell.set_facecolor("#FFFFFF" if i % 2 == 0 else "#F5F5F5")
            cell.set_edgecolor("#E0E0E0")

    ax.set_title("Summary Statistics", fontsize=14, fontweight="bold", pad=15)

    path = os.path.join(plots_dir, "08_summary_statistics_table.png")
    return save_figure(fig, path)


# ── Data Outputs ────────────────────────────────────────────────────


def _generate_ranked_csv(df: pd.DataFrame, output_dir: str) -> str:
    """Generate ranked_candidates.csv with composite score."""
    ranked = df.copy()

    # Compute composite docking score
    if "vina_score" in ranked.columns and "predicted_pIC50" in ranked.columns and "admet_score" in ranked.columns:
        vs = ranked["vina_score"]
        vina_norm = (vs - vs.max()) / (vs.min() - vs.max() + 1e-9)
        pic50 = ranked["predicted_pIC50"]
        pic50_norm = (pic50 - pic50.min()) / (pic50.max() - pic50.min() + 1e-9)
        ranked["composite_docking_score"] = (
            0.40 * vina_norm + 0.35 * pic50_norm + 0.25 * ranked["admet_score"]
        ).round(4)

    # Select and order columns
    priority_cols = [
        "rank", "record_id", "smiles", "vina_score", "composite_docking_score",
        "predicted_pIC50", "predicted_IC50_nM",
        "admet_score", "admet_class",
        "MolWt", "LogP", "QED", "RingCount",
        "admet_herg_risk", "admet_ames_risk", "admet_hepatotox_risk",
        "admet_cyp_inhibition_risk", "admet_metabolic_stability",
        "n_conformers",
    ]
    cols = [c for c in priority_cols if c in ranked.columns]
    # Add any remaining
    for c in ranked.columns:
        if c not in cols:
            cols.append(c)
    ranked = ranked[cols]

    path = os.path.join(output_dir, "ranked_candidates.csv")
    ranked.to_csv(path, index=False)
    logger.info("Ranked candidates saved: %s", path)
    return path


def _generate_enhanced_report(
    df: pd.DataFrame,
    stats: Dict,
    pop_stats: Optional[Dict],
    study_report: Optional[Dict],
    target_name: str,
    campaign_name: str,
    plots_generated: List[str],
) -> Dict:
    """Build enhanced study report dict."""
    report = {
        "study": campaign_name,
        "target": target_name,
        "generated": datetime.now().isoformat(),
        "method": "vina",
        "summary": stats,
    }

    if pop_stats:
        report["population_admet"] = pop_stats

    if study_report:
        report["original_study"] = study_report

    # Top candidates
    top_candidates = []
    for _, row in df.head(10).iterrows():
        entry = {"rank": int(row.get("rank", 0))}
        for col in ["record_id", "smiles", "vina_score", "predicted_pIC50",
                     "predicted_IC50_nM", "admet_score", "admet_class",
                     "composite_docking_score", "MolWt", "LogP", "QED"]:
            if col in row.index and pd.notna(row[col]):
                val = row[col]
                entry[col] = float(val) if isinstance(val, (np.floating, float)) else val
        top_candidates.append(entry)
    report["top_candidates"] = top_candidates
    report["plots_generated"] = [os.path.basename(p) for p in plots_generated if p]

    # Interpretation
    interp = {}
    if "vina" in stats and stats["vina"]["n"] > 0:
        v = stats["vina"]
        if v["min"] <= -5.0:
            interp["binding_quality"] = "Strong binding predicted for top candidates (Vina ≤ -5.0 kcal/mol)"
        elif v["min"] <= -3.5:
            interp["binding_quality"] = (
                f"Moderate binding predicted (best: {v['min']:.2f} kcal/mol). "
                "These are preliminary scores; re-docking with higher exhaustiveness "
                "and a fully prepared protein may improve results."
            )
        else:
            interp["binding_quality"] = (
                f"Weak binding predicted (best: {v['min']:.2f} kcal/mol). "
                "Consider optimizing the docking protocol or re-examining candidates."
            )

    if "admet" in stats:
        a = stats["admet"]
        fav = a.get("favorable", 0)
        total = fav + a.get("moderate", 0) + a.get("poor", 0)
        if total > 0:
            fav_pct = 100 * fav / total
            interp["admet_quality"] = (
                f"{fav_pct:.0f}% of docked candidates have favorable ADMET profiles "
                f"(mean score: {a.get('mean', 0):.3f})"
            )

    interp["overall_assessment"] = (
        f"Docking study screened {stats.get('vina', {}).get('n', 0)} candidates "
        f"against {target_name}. Results provide a computational ranking for "
        "prioritizing experimental validation. Further optimization through "
        "higher-exhaustiveness re-docking, MD simulations, or lead optimization "
        "is recommended for top candidates."
    )
    report["interpretation"] = interp

    return report


def _generate_summary_text(
    df: pd.DataFrame,
    stats: Dict,
    pop_stats: Optional[Dict],
    target_name: str,
    campaign_name: str,
) -> str:
    """Generate formatted text summary."""
    lines = []
    w = 60

    lines.append("=" * w)
    lines.append(f"  DrugFlow Docking Report")
    lines.append(f"  {campaign_name}")
    lines.append("=" * w)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Target: {target_name}")
    lines.append(f"Method: AutoDock Vina (CLI binary)")
    lines.append("")

    # Docking results
    lines.append("-" * w)
    lines.append("DOCKING RESULTS SUMMARY")
    lines.append("-" * w)
    if "vina" in stats and stats["vina"]["n"] > 0:
        v = stats["vina"]
        lines.append(f"  Candidates docked:     {v['n']}")
        lines.append(f"  Vina score range:      {v['min']:.3f} to {v['max']:.3f} kcal/mol")
        lines.append(f"  Mean binding energy:   {v['mean']:.3f} ± {v['std']:.3f} kcal/mol")
        lines.append(f"  Median:                {v['median']:.3f} kcal/mol")
        lines.append(f"")
        lines.append(f"  Binding tiers:")
        lines.append(f"    Strong (<= -3.5):    {v.get('n_strong', 0)}")
        lines.append(f"    Moderate (-3.0--3.5): {v.get('n_moderate', 0)}")
        lines.append(f"    Weak (> -3.0):       {v.get('n_weak', 0)}")
    lines.append("")

    # Potency
    lines.append("-" * w)
    lines.append("PREDICTED POTENCY")
    lines.append("-" * w)
    if "potency" in stats:
        p = stats["potency"]
        lines.append(f"  pIC50 range:           {p.get('min', 0):.2f} - {p.get('max', 0):.2f}")
        lines.append(f"  Mean pIC50:            {p.get('mean', 0):.3f} ± {p.get('std', 0):.3f}")
        if "ic50_nm_min" in p:
            lines.append(f"  IC50 range:            {p['ic50_nm_min']:.1f} - {p['ic50_nm_max']:.1f} nM")
    lines.append("")

    # ADMET
    lines.append("-" * w)
    lines.append("ADMET PROFILE (DOCKED SET)")
    lines.append("-" * w)
    if "admet" in stats:
        a = stats["admet"]
        total = a.get("favorable", 0) + a.get("moderate", 0) + a.get("poor", 0)
        lines.append(f"  Mean ADMET score:      {a.get('mean', 0):.3f}")
        if total > 0:
            lines.append(f"  Favorable:             {a.get('favorable', 0)} ({100*a.get('favorable', 0)/total:.1f}%)")
            lines.append(f"  Moderate:              {a.get('moderate', 0)} ({100*a.get('moderate', 0)/total:.1f}%)")
            lines.append(f"  Poor:                  {a.get('poor', 0)} ({100*a.get('poor', 0)/total:.1f}%)")
    lines.append("")

    # Population ADMET
    if pop_stats and pop_stats.get("total", 0) > 0:
        lines.append("-" * w)
        lines.append("ADMET PROFILE (ALL CANDIDATES)")
        lines.append("-" * w)
        total = pop_stats["total"]
        lines.append(f"  Total candidates:      {total}")
        lines.append(f"  Mean ADMET score:      {pop_stats.get('mean_score', 0):.3f}")
        lines.append(f"  Favorable:             {pop_stats.get('favorable', 0)} ({100*pop_stats.get('favorable', 0)/total:.1f}%)")
        lines.append(f"  Moderate:              {pop_stats.get('moderate', 0)} ({100*pop_stats.get('moderate', 0)/total:.1f}%)")
        lines.append(f"  Poor:                  {pop_stats.get('poor', 0)} ({100*pop_stats.get('poor', 0)/total:.1f}%)")
        lines.append("")

    # Safety flags
    lines.append("-" * w)
    lines.append("SAFETY RISK FLAGS")
    lines.append("-" * w)
    for name, data in stats.get("risk_flags", {}).items():
        g, y, r = data.get("green", 0), data.get("yellow", 0), data.get("red", 0)
        lines.append(f"  {name.upper():12s}  green={g}  yellow={y}  red={r}")
    lines.append("")

    # Top 10
    lines.append("-" * w)
    lines.append("TOP 10 CANDIDATES")
    lines.append("-" * w)
    header = f"  {'Rank':>4}  {'ID':>12}  {'Vina':>6}  {'pIC50':>6}  {'ADMET':>6}  {'Class':>10}  {'QED':>5}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for _, row in df.head(10).iterrows():
        rank = int(row.get("rank", 0))
        rid = str(row.get("record_id", ""))[:12]
        vina = f"{row.get('vina_score', 0):.2f}" if pd.notna(row.get("vina_score")) else "N/A"
        pic50 = f"{row.get('predicted_pIC50', 0):.2f}" if pd.notna(row.get("predicted_pIC50")) else "N/A"
        admet = f"{row.get('admet_score', 0):.3f}" if pd.notna(row.get("admet_score")) else "N/A"
        cls = str(row.get("admet_class", ""))[:10]
        qed = f"{row.get('QED', 0):.2f}" if pd.notna(row.get("QED")) else "N/A"
        lines.append(f"  {rank:>4}  {rid:>12}  {vina:>6}  {pic50:>6}  {admet:>6}  {cls:>10}  {qed:>5}")
    lines.append("")

    # Correlations
    corr = stats.get("correlations", {})
    if corr:
        lines.append("-" * w)
        lines.append("CORRELATIONS")
        lines.append("-" * w)
        for name, data in corr.items():
            lines.append(f"  {name}: Spearman rho = {data['spearman_rho']:.3f} (p = {data['p_value']:.4f})")
        lines.append("")

    lines.append("=" * w)
    lines.append("Generated by DrugFlow -- Drug Discovery Pipeline")
    lines.append("=" * w)

    return "\n".join(lines)


# ── Main Entry Point ────────────────────────────────────────────────


def generate_docking_report(
    docking_results_path: str,
    output_dir: str,
    admet_all_path: Optional[str] = None,
    study_report_path: Optional[str] = None,
    target_name: str = "Unknown Target",
    campaign_name: str = "Docking Study",
) -> str:
    """Generate a comprehensive docking results report.

    Produces 8 publication-quality PNG figures, a ranked CSV with
    composite scores, an enhanced JSON report, and a text summary.

    Args:
        docking_results_path: Path to docking_results.csv.
        output_dir: Output directory for the report.
        admet_all_path: Optional path to full ADMET CSV (all candidates).
        study_report_path: Optional path to study_report.json.
        target_name: Biological target name.
        campaign_name: Research campaign name.

    Returns:
        Path to output directory.
    """
    logger.info("Generating docking report: %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(docking_results_path)
    logger.info("Loaded %d docking results from %s", len(df), docking_results_path)

    admet_df = None
    if admet_all_path and os.path.exists(admet_all_path):
        admet_df = pd.read_csv(admet_all_path)
        logger.info("Loaded %d ADMET records from %s", len(admet_df), admet_all_path)

    study_report = None
    if study_report_path and os.path.exists(study_report_path):
        with open(study_report_path) as f:
            study_report = json.load(f)

    # Compute statistics
    stats = _compute_docking_statistics(df)
    pop_stats = _compute_population_statistics(admet_df) if admet_df is not None else None

    # Generate plots
    plots_generated = []
    plot_funcs = [
        ("Ranked candidates table", lambda: _plot_ranked_candidates_table(df, plots_dir)),
        ("Vina score distribution", lambda: _plot_vina_score_distribution(df, plots_dir)),
        ("ADMET radar chart", lambda: _plot_admet_radar_top_n(df, admet_df, plots_dir)),
        ("Score correlations", lambda: _plot_score_correlations(df, plots_dir)),
        ("ADMET class distribution", lambda: _plot_admet_class_distribution(df, admet_df, plots_dir)),
        ("Risk flag heatmap", lambda: _plot_risk_flag_heatmap(df, plots_dir)),
        ("Integrated dashboard", lambda: _plot_integrated_dashboard(df, admet_df, stats, plots_dir)),
        ("Summary statistics table", lambda: _plot_summary_statistics_table(stats, plots_dir)),
    ]

    for name, func in plot_funcs:
        try:
            path = func()
            plots_generated.append(path)
            logger.info("Generated: %s", name)
        except Exception as e:
            logger.warning("Failed to generate %s: %s", name, e)
            plots_generated.append("")

    # Generate data outputs

    # 1. Ranked CSV
    ranked_path = _generate_ranked_csv(df, output_dir)

    # 2. Enhanced JSON report
    enhanced = _generate_enhanced_report(
        df, stats, pop_stats, study_report, target_name, campaign_name, plots_generated
    )
    json_path = os.path.join(output_dir, "enhanced_study_report.json")
    with open(json_path, "w") as f:
        json.dump(enhanced, f, indent=2, default=str)
    logger.info("Enhanced report saved: %s", json_path)

    # 3. Text summary
    summary_text = _generate_summary_text(df, stats, pop_stats, target_name, campaign_name)
    txt_path = os.path.join(output_dir, "docking_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info("Summary text saved: %s", txt_path)

    # 4. Statistics CSV
    stats_rows = []
    for section, data in stats.items():
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        stats_rows.append({"section": section, "metric": f"{k}_{k2}", "value": v2})
                else:
                    stats_rows.append({"section": section, "metric": k, "value": v})
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_csv_path = os.path.join(output_dir, "docking_statistics.csv")
        stats_df.to_csv(stats_csv_path, index=False)

    n_plots = sum(1 for p in plots_generated if p)
    logger.info("Docking report complete: %d plots, 4 data files", n_plots)

    return output_dir
