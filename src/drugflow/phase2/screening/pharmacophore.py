"""Pharmacophore-based virtual screening.

Identifies molecules that contain required pharmacophore features
(hydrogen bond donors/acceptors, aromatic rings, hydrophobes, etc.)
using RDKit's chemical feature factory.
"""

from typing import Dict, List, Optional, Set

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os

from drugflow.core.constants import PHARMACOPHORE_FEATURE_FAMILIES
from drugflow.core.exceptions import ScreeningError
from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset, MoleculeRecord

logger = get_logger("screening.pharmacophore")

# Lazy-initialized feature factory
_feature_factory = None


def _get_feature_factory() -> ChemicalFeatures.BuildFeatureFactory:
    """Get or initialize the RDKit chemical feature factory."""
    global _feature_factory
    if _feature_factory is None:
        fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
    return _feature_factory


def get_pharmacophore_features(
    mol: Chem.Mol,
) -> Dict[str, int]:
    """Get counts of pharmacophore features in a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    dict
        Mapping of feature family name to count.
        e.g. {"Donor": 2, "Acceptor": 3, "Aromatic": 1}
    """
    if mol is None:
        return {}

    factory = _get_feature_factory()
    features = factory.GetFeaturesForMol(mol)

    counts: Dict[str, int] = {}
    for feat in features:
        family = feat.GetFamily()
        counts[family] = counts.get(family, 0) + 1

    return counts


def get_available_feature_families() -> List[str]:
    """Get list of available pharmacophore feature families.

    Returns
    -------
    list of str
        Available feature family names.
    """
    factory = _get_feature_factory()
    return list(factory.GetFeatureFamilies())


def check_pharmacophore_requirements(
    mol: Chem.Mol,
    required_features: Dict[str, int],
) -> Dict[str, bool]:
    """Check if molecule meets pharmacophore requirements.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to check.
    required_features : dict
        Mapping of feature family to minimum required count.
        e.g. {"Donor": 1, "Acceptor": 2}

    Returns
    -------
    dict
        Mapping of feature family to whether requirement is met.
    """
    actual = get_pharmacophore_features(mol)
    results = {}
    for family, min_count in required_features.items():
        results[family] = actual.get(family, 0) >= min_count
    return results


def screen_pharmacophore(
    dataset: MoleculeDataset,
    required_features: Dict[str, int],
    match_all: bool = True,
) -> MoleculeDataset:
    """Screen dataset for pharmacophore feature requirements.

    Stores results in rec.properties:
      - "pharm_{family}": count of that feature
      - "pharm_requirements_met": True/False

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset.
    required_features : dict
        Required feature families and minimum counts.
        e.g. {"Donor": 1, "Acceptor": 2, "Aromatic": 1}
    match_all : bool
        If True, molecule must meet ALL requirements.
        If False, meeting ANY requirement suffices.

    Returns
    -------
    MoleculeDataset
        New dataset containing only hits.

    Raises
    ------
    ScreeningError
        If required_features is empty or contains unknown families.
    """
    if not required_features:
        raise ScreeningError("required_features must not be empty")

    # Validate feature family names
    available = set(get_available_feature_families())
    for family in required_features:
        if family not in available:
            raise ScreeningError(
                f"Unknown feature family: '{family}'. "
                f"Available: {sorted(available)}"
            )

    hits = []

    for rec in progress_bar(dataset.valid_records, desc="Pharmacophore screening"):
        if rec.mol is None:
            continue

        feature_counts = get_pharmacophore_features(rec.mol)

        # Store all feature counts
        for family in PHARMACOPHORE_FEATURE_FAMILIES:
            rec.properties[f"pharm_{family}"] = feature_counts.get(family, 0)

        # Check requirements
        checks = check_pharmacophore_requirements(rec.mol, required_features)
        met_count = sum(1 for v in checks.values() if v)

        if match_all:
            is_hit = all(checks.values())
        else:
            is_hit = any(checks.values())

        rec.properties["pharm_requirements_met"] = is_hit
        rec.properties["pharm_requirements_ratio"] = (
            met_count / len(checks) if checks else 0.0
        )
        rec.add_provenance(
            f"screen:pharmacophore:{'hit' if is_hit else 'miss'}"
        )

        if is_hit:
            hits.append(rec)

    mode = "ALL" if match_all else "ANY"
    logger.info(
        f"Pharmacophore screen ({mode}): {len(hits)} hits "
        f"from {len(dataset.valid_records)} molecules"
    )

    result = MoleculeDataset(records=hits, name=f"{dataset.name}_pharm_hits")
    result._provenance = dataset._provenance + ["screen:pharmacophore"]
    return result


def parse_feature_string(feature_string: str) -> Dict[str, int]:
    """Parse a comma-separated feature requirement string.

    Format: "Donor:1,Acceptor:2,Aromatic:1" or "Donor,Acceptor,Aromatic" (count=1).

    Parameters
    ----------
    feature_string : str
        Comma-separated feature requirements.

    Returns
    -------
    dict
        Parsed requirements.
    """
    requirements = {}
    for item in feature_string.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            family, count_str = item.split(":", 1)
            requirements[family.strip()] = int(count_str.strip())
        else:
            requirements[item] = 1
    return requirements
