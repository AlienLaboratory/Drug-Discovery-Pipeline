"""Molecular standardization pipeline.

Normalizes molecules to a canonical form:
1. Remove fragments (keep largest)
2. Neutralize charges
3. Normalize functional groups
4. Canonicalize tautomers (optional)
"""

from dataclasses import dataclass, field
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from claudedd.core.logging import get_logger, progress_bar
from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus

logger = get_logger("data.standardizer")


@dataclass
class StandardizationResult:
    mol: Optional[Chem.Mol]
    original_smiles: str
    standardized_smiles: str
    changes_applied: List[str] = field(default_factory=list)
    success: bool = True
    error: str = ""


def standardize_dataset(
    dataset: MoleculeDataset,
    strip_salts: bool = True,
    neutralize: bool = True,
    normalize: bool = True,
    canonicalize_tautomers: bool = False,
    remove_stereo: bool = False,
) -> MoleculeDataset:
    """Apply standardization to all valid records in a dataset."""
    count = 0
    for rec in progress_bar(dataset.valid_records, desc="Standardizing"):
        if rec.mol is None:
            continue

        result = standardize_molecule(
            rec.mol,
            strip_salts=strip_salts,
            neutralize=neutralize,
            normalize=normalize,
            canonicalize_tautomers=canonicalize_tautomers,
            remove_stereo=remove_stereo,
        )

        if result.success and result.mol is not None:
            rec.mol = result.mol
            rec.smiles = result.standardized_smiles
            rec.status = MoleculeStatus.STANDARDIZED
            for change in result.changes_applied:
                rec.add_provenance(f"standardized:{change}")
            count += 1
        else:
            rec.add_error(f"Standardization failed: {result.error}")

    logger.info(f"Standardized {count} molecules")
    return dataset


def standardize_molecule(
    mol: Chem.Mol,
    strip_salts: bool = True,
    neutralize: bool = True,
    normalize: bool = True,
    canonicalize_tautomers: bool = False,
    remove_stereo: bool = False,
) -> StandardizationResult:
    """Standardize a single molecule."""
    original_smiles = Chem.MolToSmiles(mol, canonical=True)
    changes = []

    try:
        current_mol = Chem.RWMol(mol)

        if strip_salts:
            chooser = rdMolStandardize.LargestFragmentChooser()
            new_mol = chooser.choose(current_mol)
            if Chem.MolToSmiles(new_mol) != Chem.MolToSmiles(current_mol):
                changes.append("salt_stripped")
            current_mol = new_mol

        if normalize:
            normalizer = rdMolStandardize.Normalizer()
            new_mol = normalizer.normalize(current_mol)
            if Chem.MolToSmiles(new_mol) != Chem.MolToSmiles(current_mol):
                changes.append("normalized")
            current_mol = new_mol

        if neutralize:
            uncharger = rdMolStandardize.Uncharger()
            new_mol = uncharger.uncharge(current_mol)
            if Chem.MolToSmiles(new_mol) != Chem.MolToSmiles(current_mol):
                changes.append("neutralized")
            current_mol = new_mol

        if canonicalize_tautomers:
            canon = rdMolStandardize.TautomerCanonicalizer()
            new_mol = canon.canonicalize(current_mol)
            if Chem.MolToSmiles(new_mol) != Chem.MolToSmiles(current_mol):
                changes.append("tautomer_canonicalized")
            current_mol = new_mol

        if remove_stereo:
            Chem.RemoveStereochemistry(current_mol)
            changes.append("stereo_removed")

        standardized_smiles = Chem.MolToSmiles(current_mol, canonical=True)

        return StandardizationResult(
            mol=current_mol,
            original_smiles=original_smiles,
            standardized_smiles=standardized_smiles,
            changes_applied=changes,
            success=True,
        )

    except Exception as e:
        return StandardizationResult(
            mol=None,
            original_smiles=original_smiles,
            standardized_smiles="",
            changes_applied=changes,
            success=False,
            error=str(e),
        )
