"""Diversity-based selection for active learning.

Provides MaxMin and cluster-based diversity picking to select structurally
diverse batches of molecules for experimental testing.
"""

import logging
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def _tanimoto_distance_array(fp1: np.ndarray, fp_matrix: np.ndarray) -> np.ndarray:
    """Compute Tanimoto distance between one fingerprint and a matrix of fingerprints.

    Args:
        fp1: Single fingerprint, shape (n_bits,).
        fp_matrix: Matrix of fingerprints, shape (n_mols, n_bits).

    Returns:
        Array of distances, shape (n_mols,).
    """
    # Tanimoto similarity: |A&B| / |A|B| - |A&B|
    # For binary fingerprints: dot product / (sum_a + sum_b - dot)
    dot = fp_matrix @ fp1
    sum_a = np.sum(fp1)
    sum_b = np.sum(fp_matrix, axis=1)
    denom = sum_a + sum_b - dot
    # Avoid division by zero
    similarity = np.where(denom > 0, dot / denom, 0.0)
    return 1.0 - similarity


def maxmin_diversity_pick(
    fps: np.ndarray,
    n_pick: int,
    seed_fps: Optional[np.ndarray] = None,
    seed_idx: int = 0,
) -> List[int]:
    """MaxMin diversity picking algorithm.

    Greedily selects molecules that maximize the minimum distance to
    already-selected molecules.

    Args:
        fps: Fingerprint matrix, shape (n_mols, n_bits).
        n_pick: Number of molecules to pick.
        seed_fps: Optional pre-selected fingerprints to start from.
        seed_idx: Index of first molecule to pick (if no seed_fps).

    Returns:
        List of selected molecule indices.
    """
    n_mols = fps.shape[0]
    if n_pick >= n_mols:
        return list(range(n_mols))

    selected: List[int] = []
    remaining = set(range(n_mols))

    # Initialize min-distances
    min_dist = np.full(n_mols, np.inf)

    if seed_fps is not None and len(seed_fps) > 0:
        # Initialize from seed fingerprints
        for seed_fp in seed_fps:
            dists = _tanimoto_distance_array(seed_fp, fps)
            min_dist = np.minimum(min_dist, dists)
    else:
        # Start with a seed molecule
        if seed_idx < n_mols:
            selected.append(seed_idx)
            remaining.discard(seed_idx)
            min_dist = _tanimoto_distance_array(fps[seed_idx], fps)
            min_dist[seed_idx] = -1  # Mark as selected

    while len(selected) < n_pick and remaining:
        # Find the molecule with maximum minimum distance
        remaining_list = list(remaining)
        remaining_dists = min_dist[remaining_list]
        best_local_idx = np.argmax(remaining_dists)
        best_idx = remaining_list[best_local_idx]

        selected.append(best_idx)
        remaining.discard(best_idx)

        # Update min distances
        dists_to_new = _tanimoto_distance_array(fps[best_idx], fps)
        min_dist = np.minimum(min_dist, dists_to_new)
        min_dist[best_idx] = -1  # Mark as selected

    return selected


def cluster_diversity_pick(
    fps: np.ndarray,
    n_pick: int,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> List[int]:
    """Cluster-based diversity picking using K-Means.

    Clusters molecules and picks the one closest to each cluster centroid.

    Args:
        fps: Fingerprint matrix, shape (n_mols, n_bits).
        n_pick: Number of molecules to pick.
        n_clusters: Number of clusters. Defaults to n_pick.
        random_state: Random seed.

    Returns:
        List of selected molecule indices.
    """
    n_mols = fps.shape[0]
    if n_pick >= n_mols:
        return list(range(n_mols))

    if n_clusters is None:
        n_clusters = n_pick

    n_clusters = min(n_clusters, n_mols)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(fps)
    centroids = kmeans.cluster_centers_

    selected: List[int] = []
    for cluster_id in range(n_clusters):
        if len(selected) >= n_pick:
            break
        # Find molecule closest to centroid
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue

        cluster_fps = fps[cluster_indices]
        centroid = centroids[cluster_id]
        dists = np.sum((cluster_fps - centroid) ** 2, axis=1)
        closest_local = np.argmin(dists)
        selected.append(int(cluster_indices[closest_local]))

    return selected[:n_pick]


def compute_diversity_score(fps: np.ndarray, sample_size: int = 1000) -> float:
    """Compute internal diversity as mean pairwise Tanimoto distance.

    Args:
        fps: Fingerprint matrix, shape (n_mols, n_bits).
        sample_size: Max number of pairs to sample (for large datasets).

    Returns:
        Mean pairwise Tanimoto distance (0 = identical, 1 = maximally diverse).
    """
    n_mols = fps.shape[0]
    if n_mols < 2:
        return 0.0

    # For small datasets, compute all pairs
    if n_mols * (n_mols - 1) // 2 <= sample_size:
        total_dist = 0.0
        n_pairs = 0
        for i in range(n_mols):
            dists = _tanimoto_distance_array(fps[i], fps[i + 1:])
            total_dist += np.sum(dists)
            n_pairs += len(dists)
        return float(total_dist / n_pairs) if n_pairs > 0 else 0.0
    else:
        # Sample random pairs
        rng = np.random.RandomState(42)
        indices1 = rng.randint(0, n_mols, size=sample_size)
        indices2 = rng.randint(0, n_mols, size=sample_size)
        # Avoid same molecule
        mask = indices1 != indices2
        indices1 = indices1[mask]
        indices2 = indices2[mask]

        total_dist = 0.0
        for i, j in zip(indices1, indices2):
            dist = _tanimoto_distance_array(fps[i], fps[j:j + 1])[0]
            total_dist += dist
        return float(total_dist / len(indices1)) if len(indices1) > 0 else 0.0
