"""Benchmark runner for comparing molecular generation strategies.

Runs multiple generation strategies on the same seed dataset and collects
standardized metrics for comparison.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from claudedd.core.models import MoleculeDataset
from claudedd.phase3.benchmarking.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkStrategy:
    """Configuration for a generation strategy to benchmark."""

    name: str
    generator_fn: Callable[..., MoleculeDataset]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from running a single benchmark strategy."""

    strategy_name: str
    generated_dataset: MoleculeDataset
    metrics: Dict[str, float]
    runtime_seconds: float
    n_generated: int = 0

    def __post_init__(self):
        self.n_generated = len(self.generated_dataset)


def run_single_strategy(
    strategy: BenchmarkStrategy,
    seed_dataset: MoleculeDataset,
    n_molecules: int = 100,
    reference_dataset: Optional[MoleculeDataset] = None,
) -> BenchmarkResult:
    """Run a single generation strategy and compute metrics.

    Args:
        strategy: Strategy configuration.
        seed_dataset: Seed molecules for generation.
        n_molecules: Target number of molecules.
        reference_dataset: Optional reference for novelty calculation.

    Returns:
        BenchmarkResult with generated molecules and metrics.
    """
    logger.info(f"Running strategy: {strategy.name}")

    start_time = time.time()
    try:
        generated = strategy.generator_fn(
            seed_dataset,
            n_molecules=n_molecules,
            **strategy.params,
        )
    except Exception as e:
        logger.error(f"Strategy {strategy.name} failed: {e}")
        generated = MoleculeDataset(records=[], name=f"{strategy.name}_failed")

    runtime = time.time() - start_time

    # Extract mols for metric computation
    gen_mols = [rec.mol for rec in generated.valid_records]
    ref_mols = None
    if reference_dataset is not None:
        ref_mols = [rec.mol for rec in reference_dataset.valid_records]

    metrics = compute_all_metrics(gen_mols, ref_mols)
    metrics["runtime_seconds"] = runtime
    metrics["n_generated"] = len(gen_mols)

    result = BenchmarkResult(
        strategy_name=strategy.name,
        generated_dataset=generated,
        metrics=metrics,
        runtime_seconds=runtime,
    )

    logger.info(
        f"  {strategy.name}: {len(gen_mols)} molecules in {runtime:.1f}s, "
        f"validity={metrics.get('validity', 0):.2f}, "
        f"diversity={metrics.get('internal_diversity', 0):.2f}"
    )

    return result


def run_benchmark(
    strategies: List[BenchmarkStrategy],
    seed_dataset: MoleculeDataset,
    n_molecules: int = 100,
    reference_dataset: Optional[MoleculeDataset] = None,
) -> List[BenchmarkResult]:
    """Run multiple generation strategies and collect results.

    Args:
        strategies: List of strategies to benchmark.
        seed_dataset: Seed molecules for generation.
        n_molecules: Target molecules per strategy.
        reference_dataset: Optional reference for novelty.

    Returns:
        List of BenchmarkResults, one per strategy.
    """
    logger.info(
        f"Benchmarking {len(strategies)} strategies on "
        f"{len(seed_dataset)} seed molecules"
    )

    results = []
    for strategy in strategies:
        result = run_single_strategy(
            strategy, seed_dataset, n_molecules, reference_dataset,
        )
        results.append(result)

    logger.info(f"Benchmark complete: {len(results)} strategies evaluated")
    return results
