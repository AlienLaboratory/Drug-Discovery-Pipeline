"""Comparison and export of benchmark results.

Provides side-by-side comparison tables, strategy ranking, and
export to CSV/JSON for benchmark results.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from drugflow.phase3.benchmarking.benchmark_runner import BenchmarkResult

logger = logging.getLogger(__name__)


def compare_results(results: List[BenchmarkResult]) -> pd.DataFrame:
    """Create a side-by-side comparison DataFrame of benchmark results.

    Args:
        results: List of BenchmarkResults from run_benchmark().

    Returns:
        DataFrame with strategies as rows and metrics as columns.
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for result in results:
        row = {"strategy": result.strategy_name}
        row["n_generated"] = result.n_generated
        row["runtime_s"] = round(result.runtime_seconds, 2)
        for metric, value in sorted(result.metrics.items()):
            if metric not in ("runtime_seconds", "n_generated"):
                row[metric] = round(value, 4) if isinstance(value, float) else value
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("strategy")
    return df


def rank_strategies(
    results: List[BenchmarkResult],
    metric: str = "internal_diversity",
    higher_is_better: bool = True,
) -> List[Tuple[str, float]]:
    """Rank strategies by a specific metric.

    Args:
        results: Benchmark results.
        metric: Metric name to rank by.
        higher_is_better: If True, higher values are better.

    Returns:
        List of (strategy_name, metric_value) tuples, sorted best to worst.
    """
    rankings = []
    for result in results:
        value = result.metrics.get(metric, 0.0)
        rankings.append((result.strategy_name, float(value)))

    rankings.sort(key=lambda x: x[1], reverse=higher_is_better)
    return rankings


def compute_composite_ranking(
    results: List[BenchmarkResult],
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """Rank strategies by a weighted composite score.

    Default weights emphasize diversity and drug-likeness.

    Args:
        results: Benchmark results.
        weights: Dict of metric_name → weight. If None, uses defaults.

    Returns:
        List of (strategy_name, composite_score) tuples, sorted best first.
    """
    if weights is None:
        weights = {
            "validity": 0.15,
            "uniqueness": 0.15,
            "novelty": 0.15,
            "internal_diversity": 0.20,
            "drug_likeness_rate": 0.20,
            "mean_qed": 0.15,
        }

    rankings = []
    for result in results:
        score = 0.0
        for metric, weight in weights.items():
            value = result.metrics.get(metric, 0.0)
            score += weight * value
        rankings.append((result.strategy_name, score))

    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


def export_comparison(
    results: List[BenchmarkResult],
    output_path: str,
    format: str = "csv",
) -> str:
    """Export benchmark comparison to file.

    Args:
        results: Benchmark results.
        output_path: Output file path.
        format: "csv" or "json".

    Returns:
        Path to exported file.
    """
    if format == "csv":
        df = compare_results(results)
        df.to_csv(output_path)
        logger.info(f"Benchmark comparison exported to {output_path}")
        return output_path
    elif format == "json":
        data = []
        for result in results:
            entry = {
                "strategy": result.strategy_name,
                "n_generated": result.n_generated,
                "runtime_seconds": result.runtime_seconds,
                "metrics": result.metrics,
            }
            data.append(entry)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Benchmark comparison exported to {output_path}")
        return output_path
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")
