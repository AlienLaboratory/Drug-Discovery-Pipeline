"""Metrics for evaluating molecular generation quality.

Standard metrics from the de novo design literature: validity, uniqueness,
novelty, internal diversity, drug-likeness rate, SA score, and QED.
"""

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from claudedd.core.constants import (
    LIPINSKI_HBA_MAX,
    LIPINSKI_HBD_MAX,
    LIPINSKI_LOGP_MAX,
    LIPINSKI_MW_MAX,
)

logger = logging.getLogger(__name__)


def compute_validity_rate(mols: List[Optional[Chem.Mol]]) -> float:
    """Fraction of valid (non-None, sanitizable) molecules.

    Args:
        mols: List of RDKit molecules (may include None).

    Returns:
        Validity rate (0.0 to 1.0).
    """
    if not mols:
        return 0.0
    valid = 0
    for mol in mols:
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                valid += 1
            except Exception:
                pass
    return valid / len(mols)


def compute_uniqueness_rate(mols: List[Optional[Chem.Mol]]) -> float:
    """Fraction of unique canonical SMILES among valid molecules.

    Args:
        mols: List of RDKit molecules.

    Returns:
        Uniqueness rate (0.0 to 1.0).
    """
    valid_smiles = []
    for mol in mols:
        if mol is not None:
            try:
                smi = Chem.MolToSmiles(mol, canonical=True)
                if smi:
                    valid_smiles.append(smi)
            except Exception:
                pass

    if not valid_smiles:
        return 0.0
    return len(set(valid_smiles)) / len(valid_smiles)


def compute_novelty_rate(
    generated_mols: List[Optional[Chem.Mol]],
    reference_mols: List[Optional[Chem.Mol]],
) -> float:
    """Fraction of generated molecules not in the reference set.

    Args:
        generated_mols: Generated molecules.
        reference_mols: Reference/training molecules.

    Returns:
        Novelty rate (0.0 to 1.0).
    """
    # Build reference SMILES set
    ref_smiles: Set[str] = set()
    for mol in reference_mols:
        if mol is not None:
            try:
                smi = Chem.MolToSmiles(mol, canonical=True)
                if smi:
                    ref_smiles.add(smi)
            except Exception:
                pass

    # Check generated molecules
    n_valid = 0
    n_novel = 0
    for mol in generated_mols:
        if mol is not None:
            try:
                smi = Chem.MolToSmiles(mol, canonical=True)
                if smi:
                    n_valid += 1
                    if smi not in ref_smiles:
                        n_novel += 1
            except Exception:
                pass

    return n_novel / n_valid if n_valid > 0 else 0.0


def compute_internal_diversity(
    mols: List[Optional[Chem.Mol]],
    fp_radius: int = 2,
    fp_nbits: int = 2048,
    sample_size: int = 1000,
) -> float:
    """Mean pairwise Tanimoto distance (internal diversity).

    Args:
        mols: List of molecules.
        fp_radius: Morgan fingerprint radius.
        fp_nbits: Number of fingerprint bits.
        sample_size: Max pairs to sample for large sets.

    Returns:
        Mean Tanimoto distance (0 = identical, 1 = maximally diverse).
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    fps = []
    for mol in mols:
        if mol is not None:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
                fps.append(fp)
            except Exception:
                pass

    if len(fps) < 2:
        return 0.0

    n = len(fps)
    n_pairs = n * (n - 1) // 2

    if n_pairs <= sample_size:
        # Compute all pairs
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                total_dist += (1.0 - sim)
                count += 1
        return total_dist / count if count > 0 else 0.0
    else:
        # Sample pairs
        rng = np.random.RandomState(42)
        total_dist = 0.0
        for _ in range(sample_size):
            i, j = rng.choice(n, size=2, replace=False)
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            total_dist += (1.0 - sim)
        return total_dist / sample_size


def compute_drug_likeness_rate(mols: List[Optional[Chem.Mol]]) -> float:
    """Fraction of molecules passing Lipinski's Rule of Five.

    Args:
        mols: List of molecules.

    Returns:
        Drug-likeness rate (0.0 to 1.0).
    """
    n_valid = 0
    n_pass = 0

    for mol in mols:
        if mol is None:
            continue
        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            n_valid += 1

            violations = 0
            if mw > LIPINSKI_MW_MAX:
                violations += 1
            if logp > LIPINSKI_LOGP_MAX:
                violations += 1
            if hbd > LIPINSKI_HBD_MAX:
                violations += 1
            if hba > LIPINSKI_HBA_MAX:
                violations += 1

            if violations <= 1:
                n_pass += 1
        except Exception:
            pass

    return n_pass / n_valid if n_valid > 0 else 0.0


def compute_sa_score_distribution(mols: List[Optional[Chem.Mol]]) -> Dict[str, float]:
    """Statistics on Synthetic Accessibility scores.

    Args:
        mols: List of molecules.

    Returns:
        Dict with mean, median, std of SA scores.
    """
    from claudedd.phase2.scoring.sa_score import compute_sa_score

    scores = []
    for mol in mols:
        if mol is not None:
            try:
                score = compute_sa_score(mol)
                scores.append(score)
            except Exception:
                pass

    if not scores:
        return {"mean_sa_score": 0.0, "median_sa_score": 0.0, "std_sa_score": 0.0}

    return {
        "mean_sa_score": float(np.mean(scores)),
        "median_sa_score": float(np.median(scores)),
        "std_sa_score": float(np.std(scores)),
    }


def compute_qed_distribution(mols: List[Optional[Chem.Mol]]) -> Dict[str, float]:
    """Statistics on Quantitative Estimate of Drug-likeness (QED).

    Args:
        mols: List of molecules.

    Returns:
        Dict with mean, median, std of QED scores.
    """
    scores = []
    for mol in mols:
        if mol is not None:
            try:
                qed = QED.qed(mol)
                scores.append(qed)
            except Exception:
                pass

    if not scores:
        return {"mean_qed": 0.0, "median_qed": 0.0, "std_qed": 0.0}

    return {
        "mean_qed": float(np.mean(scores)),
        "median_qed": float(np.median(scores)),
        "std_qed": float(np.std(scores)),
    }


def compute_all_metrics(
    generated_mols: List[Optional[Chem.Mol]],
    reference_mols: Optional[List[Optional[Chem.Mol]]] = None,
) -> Dict[str, float]:
    """Compute all generation quality metrics.

    Args:
        generated_mols: List of generated molecules.
        reference_mols: Optional reference molecules for novelty.

    Returns:
        Dict of all metric name → value pairs.
    """
    metrics: Dict[str, float] = {}

    metrics["validity"] = compute_validity_rate(generated_mols)
    metrics["uniqueness"] = compute_uniqueness_rate(generated_mols)
    metrics["internal_diversity"] = compute_internal_diversity(generated_mols)
    metrics["drug_likeness_rate"] = compute_drug_likeness_rate(generated_mols)

    if reference_mols is not None:
        metrics["novelty"] = compute_novelty_rate(generated_mols, reference_mols)

    sa_stats = compute_sa_score_distribution(generated_mols)
    metrics.update(sa_stats)

    qed_stats = compute_qed_distribution(generated_mols)
    metrics.update(qed_stats)

    return metrics
