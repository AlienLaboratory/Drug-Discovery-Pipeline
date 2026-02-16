"""Tests for similarity calculations."""

import numpy as np

from drugflow.phase1.analysis.similarity import (
    tanimoto_similarity,
    dice_similarity,
    cosine_similarity,
)


def test_tanimoto_identical():
    fp = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    assert tanimoto_similarity(fp, fp) == 1.0


def test_tanimoto_different():
    fp1 = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    fp2 = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
    sim = tanimoto_similarity(fp1, fp2)
    assert 0.0 < sim < 1.0


def test_tanimoto_empty():
    fp = np.zeros(10, dtype=np.uint8)
    assert tanimoto_similarity(fp, fp) == 0.0


def test_dice_identical():
    fp = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    assert dice_similarity(fp, fp) == 1.0


def test_cosine_identical():
    fp = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    sim = cosine_similarity(fp, fp)
    assert abs(sim - 1.0) < 1e-6
