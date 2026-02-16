"""Tests for benchmark comparison and export."""

import pytest
import pandas as pd

from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.phase3.benchmarking.benchmark_runner import BenchmarkResult
from drugflow.phase3.benchmarking.comparison import (
    compare_results,
    compute_composite_ranking,
    export_comparison,
    rank_strategies,
)


@pytest.fixture
def sample_results():
    """Sample benchmark results for comparison tests."""
    ds1 = MoleculeDataset(records=[], name="gen1")
    ds2 = MoleculeDataset(records=[], name="gen2")
    return [
        BenchmarkResult(
            strategy_name="brics",
            generated_dataset=ds1,
            metrics={
                "validity": 0.9, "uniqueness": 0.8, "novelty": 0.7,
                "internal_diversity": 0.6, "drug_likeness_rate": 0.85,
                "mean_sa_score": 3.2, "mean_qed": 0.55,
            },
            runtime_seconds=1.5,
        ),
        BenchmarkResult(
            strategy_name="mutate",
            generated_dataset=ds2,
            metrics={
                "validity": 0.95, "uniqueness": 0.9, "novelty": 0.5,
                "internal_diversity": 0.4, "drug_likeness_rate": 0.9,
                "mean_sa_score": 2.5, "mean_qed": 0.65,
            },
            runtime_seconds=0.8,
        ),
    ]


def test_compare_results(sample_results):
    """Compare produces DataFrame with correct shape."""
    df = compare_results(sample_results)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "validity" in df.columns


def test_rank_strategies(sample_results):
    """Rank strategies by metric."""
    ranked = rank_strategies(sample_results, metric="validity")
    assert len(ranked) == 2
    assert ranked[0][0] == "mutate"  # 0.95 > 0.9
    assert ranked[0][1] > ranked[1][1]


def test_composite_ranking(sample_results):
    """Composite ranking combines multiple metrics."""
    ranked = compute_composite_ranking(sample_results)
    assert len(ranked) == 2
    # Both should have positive scores
    assert all(score > 0 for _, score in ranked)


def test_export_csv(sample_results, tmp_path):
    """Export comparison to CSV."""
    output = str(tmp_path / "comparison.csv")
    path = export_comparison(sample_results, output, format="csv")
    assert path == output
    df = pd.read_csv(output, index_col=0)
    assert len(df) == 2


def test_export_json(sample_results, tmp_path):
    """Export comparison to JSON."""
    import json
    output = str(tmp_path / "comparison.json")
    path = export_comparison(sample_results, output, format="json")
    assert path == output
    with open(output) as f:
        data = json.load(f)
    assert len(data) == 2
