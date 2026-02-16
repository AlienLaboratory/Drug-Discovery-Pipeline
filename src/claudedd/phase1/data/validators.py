"""Molecular validation and sanitization.

Takes RAW MoleculeRecords and promotes them to VALIDATED status
or marks them as FAILED with specific error messages.
"""

from typing import List

from rdkit import Chem
from rdkit.Chem import inchi as rdInchi

from claudedd.core.logging import get_logger, progress_bar
from claudedd.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus

logger = get_logger("data.validators")


def validate_dataset(
    dataset: MoleculeDataset,
    check_valence: bool = True,
    compute_canonical: bool = True,
    compute_inchi: bool = True,
) -> MoleculeDataset:
    """Validate all molecules in a dataset."""
    valid_count = 0
    fail_count = 0

    for rec in progress_bar(dataset.records, desc="Validating"):
        validate_molecule(
            rec,
            check_valence=check_valence,
            compute_canonical=compute_canonical,
            compute_inchi=compute_inchi,
        )
        if rec.is_valid:
            valid_count += 1
        else:
            fail_count += 1

    logger.info(f"Validation complete: {valid_count} valid, {fail_count} failed")
    return dataset


def validate_molecule(
    record: MoleculeRecord,
    check_valence: bool = True,
    compute_canonical: bool = True,
    compute_inchi: bool = True,
) -> MoleculeRecord:
    """Validate a single MoleculeRecord."""
    if record.mol is None:
        if not record.errors:
            record.add_error("Molecule is None (parsing failed)")
        return record

    mol = record.mol
    errors = check_molecular_sanity(mol, check_valence=check_valence)

    if errors:
        for err in errors:
            record.add_error(err)
        return record

    # Try sanitization
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        record.add_error(f"Sanitization failed: {e}")
        return record

    # Compute canonical SMILES
    if compute_canonical:
        try:
            record.smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            record.add_error(f"SMILES generation failed: {e}")
            return record

    # Compute InChI
    if compute_inchi:
        try:
            inchi_str = rdInchi.MolToInchi(mol)
            if inchi_str:
                record.inchi = inchi_str
                record.inchikey = rdInchi.InchiToInchiKey(inchi_str) or ""
        except Exception:
            pass  # InChI computation is non-critical

    record.status = MoleculeStatus.VALIDATED
    record.add_provenance("validated")
    return record


def check_molecular_sanity(
    mol: Chem.Mol,
    check_valence: bool = True,
) -> List[str]:
    """Run sanity checks on an RDKit Mol. Returns list of error strings."""
    errors = []

    if mol is None:
        errors.append("Molecule is None")
        return errors

    # Check heavy atom count
    if mol.GetNumHeavyAtoms() == 0:
        errors.append("No heavy atoms")

    # Check for valence errors
    if check_valence:
        try:
            Chem.SanitizeMol(mol)
        except Chem.AtomValenceException as e:
            errors.append(f"Valence error: {e}")
        except Chem.KekulizeException as e:
            errors.append(f"Kekulization error: {e}")
        except Exception as e:
            errors.append(f"Sanitization error: {e}")

    # Check molecular weight range (sanity)
    try:
        from rdkit.Chem import Descriptors
        mw = Descriptors.MolWt(mol)
        if mw < 10:
            errors.append(f"Molecular weight too low: {mw:.1f}")
        elif mw > 5000:
            errors.append(f"Molecular weight too high: {mw:.1f}")
    except Exception:
        pass

    return errors
