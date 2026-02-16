"""Tests for the benchmark runner."""

import pytest
from rdkit import Chem

from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.phase3.benchmarking.benchmark_runner import (
    BenchmarkResult,
    BenchmarkStrategy,
    run_benchmark,
    run_single_strategy,
)


def _dummy_generator(dataset, n_molecules=10, seed=42):
    """Simple generator that returns copies of input molecules."""
    records = []
    for i, rec in enumerate(dataset.valid_records[:n_molecules]):
        new_rec = MoleculeRecord(
            mol=rec.mol, smiles=rec.canonical_smiles,
            source_id=f"gen_{i}", status=MoleculeStatus.RAW,
        )
        records.append(new_rec)
    return MoleculeDataset(records=records, name="dummy_gen")


def test_benchmark_strategy_creation():
    """BenchmarkStrategy dataclass works."""
    strategy = BenchmarkStrategy(
        name="test", generator_fn=_dummy_generator,
    )
    assert strategy.name == "test"


def test_run_single_strategy(seed_dataset):
    """Run single strategy produces result."""
    strategy = BenchmarkStrategy(
        name="dummy", generator_fn=_dummy_generator,
    )
    result = run_single_strategy(strategy, seed_dataset, n_molecules=5)
    assert isinstance(result, BenchmarkResult)
    assert result.strategy_name == "dummy"
    assert result.n_generated > 0
    assert "validity" in result.metrics
    assert result.runtime_seconds >= 0


def test_run_benchmark_multiple(seed_dataset):
    """Run multiple strategies returns results for each."""
    strategies = [
        BenchmarkStrategy(name="dummy1", generator_fn=_dummy_generator),
        BenchmarkStrategy(name="dummy2", generator_fn=_dummy_generator),
    ]
    results = run_benchmark(strategies, seed_dataset, n_molecules=3)
    assert len(results) == 2
    assert results[0].strategy_name == "dummy1"
    assert results[1].strategy_name == "dummy2"


def test_run_benchmark_with_reference(seed_dataset):
    """Benchmark with reference dataset includes novelty."""
    strategy = BenchmarkStrategy(
        name="dummy", generator_fn=_dummy_generator,
    )
    results = run_benchmark(
        [strategy], seed_dataset, n_molecules=3,
        reference_dataset=seed_dataset,
    )
    assert "novelty" in results[0].metrics
