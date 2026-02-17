"""Research report generator for drug discovery campaigns.

Assembles comprehensive results packages with candidate data,
statistics, model information, and visualizations for analysis
and publication.
"""

import csv
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_research_report(
    candidates_path: str,
    training_data_path: str,
    output_dir: str,
    model_path: Optional[str] = None,
    model_comparison_path: Optional[str] = None,
    target_name: str = "Unknown Target",
    campaign_name: str = "Research Campaign",
) -> str:
    """Generate a comprehensive research report folder.

    Creates an output directory with summary, CSVs, and plots
    summarizing a drug discovery campaign's results.

    Parameters
    ----------
    candidates_path : str
        Path to CSV with novel candidate molecules and predictions.
    training_data_path : str
        Path to CSV with original curated training data.
    output_dir : str
        Output directory for the report.
    model_path : str, optional
        Path to trained QSAR model (.joblib) for metadata.
    model_comparison_path : str, optional
        Path to model comparison CSV.
    target_name : str
        Name of the biological target (e.g., "BCL-2").
    campaign_name : str
        Name of the research campaign.

    Returns
    -------
    str
        Path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    candidates_df = pd.read_csv(candidates_path)
    training_df = pd.read_csv(training_data_path)

    # Load model info if provided
    model_info = None
    if model_path and os.path.exists(model_path):
        model_info = _load_model_info(model_path)

    # Load comparison if provided
    comparison_df = None
    if model_comparison_path and os.path.exists(model_comparison_path):
        comparison_df = pd.read_csv(model_comparison_path)

    # Compute candidate statistics
    cand_stats = _compute_candidate_stats(candidates_df)
    train_stats = _compute_candidate_stats(training_df)

    # Generate summary text
    summary_text = _generate_summary_text(
        candidates_df=candidates_df,
        training_df=training_df,
        cand_stats=cand_stats,
        train_stats=train_stats,
        model_info=model_info,
        comparison_df=comparison_df,
        target_name=target_name,
        campaign_name=campaign_name,
    )
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info(f"Summary written to {summary_path}")

    # Export candidates CSVs
    candidates_df.to_csv(
        os.path.join(output_dir, "candidates_full.csv"), index=False,
    )

    # Top 50 by predicted potency
    sort_col = _find_potency_column(candidates_df)
    if sort_col:
        top50 = candidates_df.nlargest(50, sort_col)
    else:
        top50 = candidates_df.head(50)
    top50.to_csv(
        os.path.join(output_dir, "candidates_top50.csv"), index=False,
    )

    # Training data stats CSV
    _export_stats_csv(train_stats, os.path.join(output_dir, "training_data_stats.csv"))

    # Copy comparison if exists
    if comparison_df is not None:
        comparison_df.to_csv(
            os.path.join(output_dir, "model_comparison.csv"), index=False,
        )

    # Generate plots
    try:
        _plot_property_distributions(candidates_df, plots_dir)
    except Exception as e:
        logger.warning(f"Property distributions plot failed: {e}")

    try:
        _plot_potency_distribution(candidates_df, plots_dir)
    except Exception as e:
        logger.warning(f"Potency distribution plot failed: {e}")

    try:
        _plot_chemical_space(candidates_df, training_df, plots_dir)
    except Exception as e:
        logger.warning(f"Chemical space plot failed: {e}")

    try:
        _plot_score_distribution(candidates_df, plots_dir)
    except Exception as e:
        logger.warning(f"Score distribution plot failed: {e}")

    logger.info(f"Research report generated at {output_dir}")
    return output_dir


def _find_potency_column(df: pd.DataFrame) -> Optional[str]:
    """Find the most likely potency/score column."""
    for col in ["predicted_pIC50", "pIC50", "predicted_activity",
                "composite_score", "potency_score"]:
        if col in df.columns:
            return col
    return None


def _load_model_info(model_path: str) -> Dict[str, Any]:
    """Load model metadata from a joblib file."""
    try:
        import joblib
        data = joblib.load(model_path)
        if hasattr(data, "summary"):
            return data.summary()
        elif isinstance(data, dict):
            return {
                "model_type": data.get("model_type", "unknown"),
                "task": data.get("task", "unknown"),
                "n_features": data.get("n_features", 0),
                "training_metrics": data.get("training_metrics", {}),
                "metadata": data.get("metadata", {}),
            }
        return {}
    except Exception as e:
        logger.warning(f"Could not load model info: {e}")
        return {}


def _compute_candidate_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute aggregate statistics for a molecule DataFrame."""
    stats = {"n_molecules": len(df)}

    numeric_cols = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "QED",
                    "predicted_pIC50", "pIC50", "predicted_activity",
                    "composite_score", "confidence_score",
                    "AromaticRings", "RingCount", "NumRotatableBonds",
                    "MolMR", "HeavyAtoms", "FrCSP3"]

    for col in numeric_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                stats[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "median": float(vals.median()),
                    "count": int(len(vals)),
                }

    return stats


def _export_stats_csv(stats: Dict[str, Any], path: str) -> None:
    """Export statistics dict to CSV."""
    rows = []
    for key, val in stats.items():
        if key == "n_molecules":
            rows.append({"property": "n_molecules", "value": val})
        elif isinstance(val, dict):
            for metric, v in val.items():
                rows.append({
                    "property": f"{key}_{metric}",
                    "value": round(v, 4) if isinstance(v, float) else v,
                })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["property", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _generate_summary_text(
    candidates_df: pd.DataFrame,
    training_df: pd.DataFrame,
    cand_stats: Dict[str, Any],
    train_stats: Dict[str, Any],
    model_info: Optional[Dict[str, Any]],
    comparison_df: Optional[pd.DataFrame],
    target_name: str,
    campaign_name: str,
) -> str:
    """Generate human-readable summary text."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  DrugFlow Research Report")
    lines.append(f"  {campaign_name}")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Target: {target_name}")
    lines.append("")

    # Dataset overview
    lines.append("-" * 50)
    lines.append("DATASET OVERVIEW")
    lines.append("-" * 50)
    lines.append(f"Training molecules:  {len(training_df)}")
    lines.append(f"Novel candidates:    {len(candidates_df)}")
    lines.append("")

    # Training data stats
    lines.append("-" * 50)
    lines.append("TRAINING DATA PROPERTIES")
    lines.append("-" * 50)
    _append_property_stats(lines, train_stats)

    # Candidate stats
    lines.append("-" * 50)
    lines.append("CANDIDATE PROPERTIES")
    lines.append("-" * 50)
    _append_property_stats(lines, cand_stats)

    # Potency summary
    potency_col = _find_potency_column(candidates_df)
    if potency_col and potency_col in cand_stats:
        ps = cand_stats[potency_col]
        lines.append("")
        lines.append("-" * 50)
        lines.append("PREDICTED POTENCY")
        lines.append("-" * 50)
        lines.append(f"Column: {potency_col}")
        lines.append(f"Mean:   {ps['mean']:.3f}")
        lines.append(f"Median: {ps['median']:.3f}")
        lines.append(f"Range:  {ps['min']:.3f} - {ps['max']:.3f}")
        lines.append(f"Std:    {ps['std']:.3f}")

        # Count by potency bins
        vals = pd.to_numeric(candidates_df[potency_col], errors="coerce").dropna()
        if "pIC50" in potency_col.lower() or potency_col == "pIC50":
            n_sub_nm = int((vals >= 9.0).sum())
            n_potent = int((vals >= 8.0).sum())
            n_moderate = int(((vals >= 7.0) & (vals < 8.0)).sum())
            lines.append(f"Sub-nanomolar (pIC50 >= 9): {n_sub_nm}")
            lines.append(f"Very potent (pIC50 >= 8):   {n_potent}")
            lines.append(f"Potent (pIC50 7-8):         {n_moderate}")

    # Model info
    if model_info:
        lines.append("")
        lines.append("-" * 50)
        lines.append("QSAR MODEL")
        lines.append("-" * 50)
        lines.append(f"Type: {model_info.get('model_type', 'unknown')}")
        lines.append(f"Task: {model_info.get('task', 'unknown')}")
        lines.append(f"Features: {model_info.get('n_features', '?')}")
        tm = model_info.get("training_metrics", {})
        for k, v in tm.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")

    # Model comparison
    if comparison_df is not None and len(comparison_df) > 0:
        lines.append("")
        lines.append("-" * 50)
        lines.append("MODEL COMPARISON")
        lines.append("-" * 50)
        for _, row in comparison_df.iterrows():
            mt = row.get("model_type", "?")
            cv_r2 = row.get("cv_r2", "?")
            test_r2 = row.get("test_r2", "?")
            try:
                cv_r2 = f"{float(cv_r2):.4f}"
            except (ValueError, TypeError):
                pass
            try:
                test_r2 = f"{float(test_r2):.4f}"
            except (ValueError, TypeError):
                pass
            lines.append(f"  {mt}: CV R2={cv_r2}, Test R2={test_r2}")

    # Top candidates
    lines.append("")
    lines.append("-" * 50)
    lines.append("TOP 10 CANDIDATES")
    lines.append("-" * 50)
    sort_col = _find_potency_column(candidates_df)
    if sort_col:
        top10 = candidates_df.nlargest(10, sort_col)
    else:
        top10 = candidates_df.head(10)

    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        smi = str(row.get("smiles", ""))[:60]
        pred = ""
        if sort_col and sort_col in row:
            try:
                pred = f" | {sort_col}={float(row[sort_col]):.3f}"
            except (ValueError, TypeError):
                pass
        lines.append(f"  {rank:>2}. {smi}...{pred}")

    # Footer
    lines.append("")
    lines.append("=" * 70)
    lines.append("Report generated by DrugFlow")
    lines.append(f"https://github.com/AlienLaboratory/Drug-Discovery-Pipeline")
    lines.append("=" * 70)

    return "\n".join(lines)


def _append_property_stats(lines: list, stats: Dict[str, Any]) -> None:
    """Append formatted property statistics to lines."""
    props_to_show = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "QED",
                     "RingCount", "AromaticRings"]
    for prop in props_to_show:
        if prop in stats and isinstance(stats[prop], dict):
            s = stats[prop]
            lines.append(
                f"  {prop:<18}: {s['mean']:>8.2f} +/- {s['std']:<8.2f} "
                f"[{s['min']:.1f} - {s['max']:.1f}]"
            )


# ── Plot generation ─────────────────────────────────────────


def _plot_property_distributions(df: pd.DataFrame, plots_dir: str) -> str:
    """Plot property distribution histograms for candidates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    props = []
    prop_names = []
    for col, name in [
        ("MolWt", "Molecular Weight (Da)"),
        ("LogP", "LogP"),
        ("TPSA", "TPSA"),
        ("QED", "QED Score"),
        ("HBD", "H-Bond Donors"),
        ("HBA", "H-Bond Acceptors"),
        ("RingCount", "Ring Count"),
        ("AromaticRings", "Aromatic Rings"),
        ("FrCSP3", "Fraction sp3"),
    ]:
        if col in df.columns:
            props.append(col)
            prop_names.append(name)

    if not props:
        return ""

    n = len(props)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (col, name) in enumerate(zip(props, prop_names)):
        ax = axes[i]
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(vals, bins=30, color="#2196F3", alpha=0.8, edgecolor="white")
        ax.set_title(name, fontsize=11)
        ax.set_ylabel("Count")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(plots_dir, "property_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_potency_distribution(df: pd.DataFrame, plots_dir: str) -> str:
    """Plot predicted potency histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = _find_potency_column(df)
    if not col:
        return ""

    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(vals) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=40, color="#4CAF50", alpha=0.8, edgecolor="white")
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Predicted Potency Distribution ({col})", fontsize=14)
    ax.axvline(vals.median(), color="red", linestyle="--", linewidth=2,
               label=f"Median: {vals.median():.2f}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(plots_dir, "potency_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_chemical_space(
    candidates_df: pd.DataFrame,
    training_df: pd.DataFrame,
    plots_dir: str,
) -> str:
    """Plot PCA chemical space of candidates vs training data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.decomposition import PCA

    def _get_fps(df, max_n=500):
        fps = []
        smi_col = None
        for c in ["smiles", "canonical_smiles", "SMILES"]:
            if c in df.columns:
                smi_col = c
                break
        if smi_col is None:
            return np.array([])

        for smi in df[smi_col].head(max_n):
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fps.append(np.array(fp))
        return np.array(fps) if fps else np.array([])

    # Compute fingerprints
    train_fps = _get_fps(training_df, max_n=500)
    cand_fps = _get_fps(candidates_df, max_n=500)

    if len(train_fps) < 5 or len(cand_fps) < 5:
        return ""

    # PCA on combined data
    combined = np.vstack([train_fps, cand_fps])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(combined)

    n_train = len(train_fps)
    train_coords = coords[:n_train]
    cand_coords = coords[n_train:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(train_coords[:, 0], train_coords[:, 1],
               c="#90CAF9", alpha=0.4, s=20, label="Training data")
    ax.scatter(cand_coords[:, 0], cand_coords[:, 1],
               c="#F44336", alpha=0.7, s=30, label="Novel candidates")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.set_title("Chemical Space: Candidates vs Training Data", fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(plots_dir, "chemical_space_pca.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_score_distribution(df: pd.DataFrame, plots_dir: str) -> str:
    """Plot composite/potency score distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Try multiple score columns
    score_cols = []
    for col in ["predicted_pIC50", "composite_score", "predicted_activity"]:
        if col in df.columns:
            score_cols.append(col)

    if not score_cols:
        return ""

    fig, axes = plt.subplots(1, len(score_cols),
                             figsize=(6 * len(score_cols), 5))
    if len(score_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, score_cols):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=30, color="#FF9800", alpha=0.8, edgecolor="white")
        ax.set_title(col, fontsize=12)
        ax.set_ylabel("Count")
        ax.axvline(vals.mean(), color="red", linestyle="--",
                   label=f"Mean: {vals.mean():.2f}")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(plots_dir, "score_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
