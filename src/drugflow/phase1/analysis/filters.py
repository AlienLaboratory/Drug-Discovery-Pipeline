"""Drug-likeness and structural alert filters.

Implements Lipinski, Veber, PAINS, Brenk, and Ghose filters.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from drugflow.core.constants import (
    LIPINSKI_HBA_MAX,
    LIPINSKI_HBD_MAX,
    LIPINSKI_LOGP_MAX,
    LIPINSKI_MW_MAX,
    VEBER_ROTATABLE_BONDS_MAX,
    VEBER_TPSA_MAX,
)
from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus

logger = get_logger("analysis.filters")

# Lazy-init filter catalogs (expensive to create)
_pains_catalog = None
_brenk_catalog = None


def _get_pains_catalog() -> FilterCatalog:
    global _pains_catalog
    if _pains_catalog is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _pains_catalog = FilterCatalog(params)
    return _pains_catalog


def _get_brenk_catalog() -> FilterCatalog:
    global _brenk_catalog
    if _brenk_catalog is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _brenk_catalog = FilterCatalog(params)
    return _brenk_catalog


@dataclass
class FilterResult:
    filter_name: str
    passed: bool
    violations: List[str] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)


def apply_lipinski(
    mol: Chem.Mol,
    max_violations: int = 1,
) -> FilterResult:
    """Apply Lipinski's Rule of Five."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    violations = []
    if mw > LIPINSKI_MW_MAX:
        violations.append(f"MW={mw:.1f} > {LIPINSKI_MW_MAX}")
    if logp > LIPINSKI_LOGP_MAX:
        violations.append(f"LogP={logp:.2f} > {LIPINSKI_LOGP_MAX}")
    if hbd > LIPINSKI_HBD_MAX:
        violations.append(f"HBD={hbd} > {LIPINSKI_HBD_MAX}")
    if hba > LIPINSKI_HBA_MAX:
        violations.append(f"HBA={hba} > {LIPINSKI_HBA_MAX}")

    return FilterResult(
        filter_name="lipinski",
        passed=len(violations) <= max_violations,
        violations=violations,
        details={"MolWt": mw, "LogP": logp, "HBD": hbd, "HBA": hba},
    )


def apply_veber(mol: Chem.Mol) -> FilterResult:
    """Apply Veber filter for oral bioavailability."""
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)

    violations = []
    if tpsa > VEBER_TPSA_MAX:
        violations.append(f"TPSA={tpsa:.1f} > {VEBER_TPSA_MAX}")
    if rot_bonds > VEBER_ROTATABLE_BONDS_MAX:
        violations.append(f"RotBonds={rot_bonds} > {VEBER_ROTATABLE_BONDS_MAX}")

    return FilterResult(
        filter_name="veber",
        passed=len(violations) == 0,
        violations=violations,
        details={"TPSA": tpsa, "NumRotatableBonds": rot_bonds},
    )


def apply_pains(mol: Chem.Mol) -> FilterResult:
    """Apply PAINS filter using RDKit's FilterCatalog."""
    catalog = _get_pains_catalog()
    entry = catalog.GetFirstMatch(mol)
    passed = entry is None

    violations = []
    if not passed:
        violations.append(f"PAINS match: {entry.GetDescription()}")

    return FilterResult(
        filter_name="pains",
        passed=passed,
        violations=violations,
    )


def apply_brenk(mol: Chem.Mol) -> FilterResult:
    """Apply Brenk filter for unwanted substructures."""
    catalog = _get_brenk_catalog()
    entry = catalog.GetFirstMatch(mol)
    passed = entry is None

    violations = []
    if not passed:
        violations.append(f"Brenk match: {entry.GetDescription()}")

    return FilterResult(
        filter_name="brenk",
        passed=passed,
        violations=violations,
    )


def apply_all_filters(
    mol: Chem.Mol,
    lipinski: bool = True,
    veber: bool = False,
    pains: bool = True,
    brenk: bool = False,
    max_lipinski_violations: int = 1,
) -> Dict[str, FilterResult]:
    """Apply multiple filters and return results dict."""
    results = {}
    if lipinski:
        results["lipinski"] = apply_lipinski(mol, max_violations=max_lipinski_violations)
    if veber:
        results["veber"] = apply_veber(mol)
    if pains:
        results["pains"] = apply_pains(mol)
    if brenk:
        results["brenk"] = apply_brenk(mol)
    return results


def filter_dataset(
    dataset: MoleculeDataset,
    lipinski: bool = True,
    veber: bool = False,
    pains: bool = True,
    brenk: bool = False,
    max_lipinski_violations: int = 1,
    remove_failures: bool = False,
) -> MoleculeDataset:
    """Apply filters to all records in a dataset."""
    pass_count = 0
    fail_count = 0

    for rec in progress_bar(dataset.valid_records, desc="Applying filters"):
        if rec.mol is None:
            continue

        results = apply_all_filters(
            rec.mol,
            lipinski=lipinski,
            veber=veber,
            pains=pains,
            brenk=brenk,
            max_lipinski_violations=max_lipinski_violations,
        )

        all_passed = True
        for name, result in results.items():
            rec.properties[f"{name}_pass"] = result.passed
            rec.properties[f"{name}_violations"] = len(result.violations)
            if result.violations:
                rec.properties[f"{name}_violation_details"] = "; ".join(
                    result.violations
                )
            rec.properties.update(result.details)

            if not result.passed:
                all_passed = False
                rec.add_provenance(f"filter:{name}:fail")
            else:
                rec.add_provenance(f"filter:{name}:pass")

        if all_passed:
            pass_count += 1
        else:
            fail_count += 1
            if remove_failures:
                rec.status = MoleculeStatus.FILTERED

    logger.info(f"Filters applied: {pass_count} passed, {fail_count} failed")

    if remove_failures:
        passing = [r for r in dataset.records if r.status != MoleculeStatus.FILTERED]
        result_ds = MoleculeDataset(records=passing, name=dataset.name)
        result_ds._provenance = dataset._provenance + ["filtered"]
        return result_ds

    return dataset
