"""Molecular similarity calculations.

Operates on pre-computed fingerprints.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset

logger = get_logger("analysis.similarity")


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two fingerprint arrays."""
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def dice_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Dice similarity between two fingerprint arrays."""
    intersection = np.sum(fp1 & fp2)
    total = np.sum(fp1) + np.sum(fp2)
    if total == 0:
        return 0.0
    return 2.0 * float(intersection) / float(total)


def cosine_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute cosine similarity between two fingerprint arrays."""
    dot = np.dot(fp1.astype(float), fp2.astype(float))
    norm1 = np.linalg.norm(fp1.astype(float))
    norm2 = np.linalg.norm(fp2.astype(float))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


_METRICS = {
    "tanimoto": tanimoto_similarity,
    "dice": dice_similarity,
    "cosine": cosine_similarity,
}


def get_similarity_func(metric: str = "tanimoto"):
    func = _METRICS.get(metric)
    if func is None:
        raise ValueError(
            f"Unknown metric: {metric}. Available: {list(_METRICS.keys())}"
        )
    return func


def pairwise_similarity_matrix(
    dataset: MoleculeDataset,
    fp_type: str = "morgan_r2_2048",
    metric: str = "tanimoto",
) -> np.ndarray:
    """Compute NxN pairwise similarity matrix."""
    sim_func = get_similarity_func(metric)
    valid = [r for r in dataset.valid_records if fp_type in r.fingerprints]

    n = len(valid)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in progress_bar(range(n), desc="Computing similarity matrix"):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = sim_func(valid[i].fingerprints[fp_type],
                           valid[j].fingerprints[fp_type])
            matrix[i, j] = sim
            matrix[j, i] = sim

    return matrix


def find_nearest_neighbors(
    query: np.ndarray,
    dataset: MoleculeDataset,
    fp_type: str = "morgan_r2_2048",
    top_k: int = 10,
    metric: str = "tanimoto",
) -> List[Tuple[int, float, str]]:
    """Find top-K most similar molecules to a query fingerprint.

    Returns list of (record_index, similarity_score, smiles) tuples.
    """
    sim_func = get_similarity_func(metric)
    valid = [(i, r) for i, r in enumerate(dataset.records)
             if r.is_valid and fp_type in r.fingerprints]

    similarities = []
    for idx, rec in valid:
        sim = sim_func(query, rec.fingerprints[fp_type])
        similarities.append((idx, sim, rec.canonical_smiles))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def diversity_analysis(
    dataset: MoleculeDataset,
    fp_type: str = "morgan_r2_2048",
    sample_size: int = 1000,
) -> Dict[str, float]:
    """Compute diversity statistics for a dataset."""
    valid = [r for r in dataset.valid_records if fp_type in r.fingerprints]

    if len(valid) < 2:
        return {
            "mean_similarity": 0.0,
            "median_similarity": 0.0,
            "min_similarity": 0.0,
            "max_similarity": 0.0,
            "std_similarity": 0.0,
            "diversity_score": 1.0,
            "n_molecules": len(valid),
        }

    # Sample if dataset is large
    if len(valid) > sample_size:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(valid), size=sample_size, replace=False)
        valid = [valid[i] for i in indices]

    # Compute pairwise similarities (upper triangle only)
    sims = []
    n = len(valid)
    for i in range(n):
        for j in range(i + 1, n):
            sim = tanimoto_similarity(
                valid[i].fingerprints[fp_type],
                valid[j].fingerprints[fp_type],
            )
            sims.append(sim)

    sims = np.array(sims)

    return {
        "mean_similarity": float(np.mean(sims)),
        "median_similarity": float(np.median(sims)),
        "min_similarity": float(np.min(sims)),
        "max_similarity": float(np.max(sims)),
        "std_similarity": float(np.std(sims)),
        "diversity_score": 1.0 - float(np.mean(sims)),
        "n_molecules": n,
    }
