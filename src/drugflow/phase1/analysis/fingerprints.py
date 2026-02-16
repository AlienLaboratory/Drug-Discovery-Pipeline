"""Molecular fingerprint generation.

Supports Morgan (ECFP), MACCS, RDKit topological, AtomPair,
and TopologicalTorsion fingerprints.
"""

from typing import Dict, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator

from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset

logger = get_logger("analysis.fingerprints")


def compute_fingerprints_dataset(
    dataset: MoleculeDataset,
    fp_types: Optional[Dict[str, Dict]] = None,
) -> MoleculeDataset:
    """Compute fingerprints for all valid records."""
    if fp_types is None:
        fp_types = {
            "morgan_r2_2048": {"type": "morgan", "radius": 2, "nbits": 2048},
            "maccs": {"type": "maccs"},
        }

    count = 0
    for rec in progress_bar(dataset.valid_records, desc="Computing fingerprints"):
        if rec.mol is None:
            continue
        try:
            for fp_name, params in fp_types.items():
                fp_type = params.get("type", fp_name)
                fp = _compute_single_fp(rec.mol, fp_type, params)
                if fp is not None:
                    rec.fingerprints[fp_name] = fp
            rec.add_provenance("fingerprints:computed")
            count += 1
        except Exception as e:
            logger.warning(
                f"Fingerprint computation failed for {rec.record_id}: {e}"
            )

    logger.info(f"Computed fingerprints for {count} molecules")
    return dataset


def _compute_single_fp(
    mol: Chem.Mol,
    fp_type: str,
    params: dict,
) -> Optional[np.ndarray]:
    """Route to the appropriate fingerprint function."""
    dispatch = {
        "morgan": lambda: compute_morgan(
            mol,
            radius=params.get("radius", 2),
            nbits=params.get("nbits", 2048),
        ),
        "maccs": lambda: compute_maccs(mol),
        "rdkit": lambda: compute_rdkit_fp(
            mol,
            nbits=params.get("nbits", 2048),
        ),
        "atom_pair": lambda: compute_atom_pair(
            mol,
            nbits=params.get("nbits", 2048),
        ),
        "topological_torsion": lambda: compute_topological_torsion(
            mol,
            nbits=params.get("nbits", 2048),
        ),
    }
    func = dispatch.get(fp_type)
    if func is None:
        logger.warning(f"Unknown fingerprint type: {fp_type}")
        return None
    return func()


def compute_morgan(
    mol: Chem.Mol,
    radius: int = 2,
    nbits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprint (ECFP-equivalent)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=nbits
    )
    fp = gen.GetFingerprint(mol)
    return bitvect_to_numpy(fp)


def compute_maccs(mol: Chem.Mol) -> np.ndarray:
    """Compute MACCS 166-key fingerprint."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return bitvect_to_numpy(fp)


def compute_rdkit_fp(
    mol: Chem.Mol,
    min_path: int = 1,
    max_path: int = 7,
    nbits: int = 2048,
) -> np.ndarray:
    """Compute RDKit topological fingerprint."""
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(
        minPath=min_path, maxPath=max_path, fpSize=nbits
    )
    fp = gen.GetFingerprint(mol)
    return bitvect_to_numpy(fp)


def compute_atom_pair(mol: Chem.Mol, nbits: int = 2048) -> np.ndarray:
    """Compute Atom Pair fingerprint."""
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nbits)
    fp = gen.GetFingerprint(mol)
    return bitvect_to_numpy(fp)


def compute_topological_torsion(mol: Chem.Mol, nbits: int = 2048) -> np.ndarray:
    """Compute Topological Torsion fingerprint."""
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nbits)
    fp = gen.GetFingerprint(mol)
    return bitvect_to_numpy(fp)


def bitvect_to_numpy(bitvect) -> np.ndarray:
    """Convert RDKit BitVect to numpy uint8 array."""
    arr = np.zeros(bitvect.GetNumBits(), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr
