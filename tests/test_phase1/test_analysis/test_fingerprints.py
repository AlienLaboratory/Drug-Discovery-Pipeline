"""Tests for fingerprint generation."""

import numpy as np
from rdkit import Chem

from drugflow.phase1.analysis.fingerprints import (
    compute_maccs,
    compute_morgan,
    compute_rdkit_fp,
)


def test_morgan_fingerprint_shape(aspirin_mol):
    fp = compute_morgan(aspirin_mol)
    assert fp.shape == (2048,)
    assert fp.dtype == np.uint8


def test_morgan_fingerprint_nonzero(aspirin_mol):
    fp = compute_morgan(aspirin_mol)
    assert np.sum(fp) > 0


def test_maccs_fingerprint_shape(aspirin_mol):
    fp = compute_maccs(aspirin_mol)
    assert fp.shape == (167,)


def test_rdkit_fingerprint_shape(aspirin_mol):
    fp = compute_rdkit_fp(aspirin_mol)
    assert fp.shape == (2048,)


def test_fingerprints_reproducible(aspirin_mol):
    fp1 = compute_morgan(aspirin_mol)
    fp2 = compute_morgan(aspirin_mol)
    assert np.array_equal(fp1, fp2)
