"""Tests for diversity-based selection."""

import numpy as np
import pytest

from drugflow.phase3.active_learning.diversity import (
    cluster_diversity_pick,
    compute_diversity_score,
    maxmin_diversity_pick,
)


@pytest.fixture
def sample_fps():
    """Sample fingerprint matrix (10 x 64 bits)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 2, size=(10, 64)).astype(float)


def test_maxmin_pick_basic(sample_fps):
    """MaxMin picks correct number of molecules."""
    selected = maxmin_diversity_pick(sample_fps, n_pick=3)
    assert len(selected) == 3
    assert len(set(selected)) == 3  # All unique


def test_maxmin_pick_more_than_available(sample_fps):
    """Requesting more than available returns all."""
    selected = maxmin_diversity_pick(sample_fps, n_pick=100)
    assert len(selected) == 10


def test_maxmin_pick_with_seed(sample_fps):
    """MaxMin with seed fingerprints."""
    seed_fps = sample_fps[:2]
    selected = maxmin_diversity_pick(sample_fps, n_pick=3, seed_fps=seed_fps)
    assert len(selected) == 3


def test_cluster_diversity_pick(sample_fps):
    """Cluster-based picking selects correct number."""
    selected = cluster_diversity_pick(sample_fps, n_pick=3)
    assert len(selected) == 3
    assert len(set(selected)) == 3


def test_cluster_pick_more_than_available(sample_fps):
    """Requesting more than available returns all."""
    selected = cluster_diversity_pick(sample_fps, n_pick=100)
    assert len(selected) == 10


def test_diversity_score(sample_fps):
    """Diversity score is between 0 and 1."""
    score = compute_diversity_score(sample_fps)
    assert 0.0 <= score <= 1.0


def test_diversity_score_identical():
    """Identical fingerprints have zero diversity."""
    fps = np.ones((5, 64))
    score = compute_diversity_score(fps)
    assert score == pytest.approx(0.0, abs=0.01)


def test_diversity_score_single():
    """Single molecule has zero diversity."""
    fps = np.random.rand(1, 64)
    score = compute_diversity_score(fps)
    assert score == 0.0
