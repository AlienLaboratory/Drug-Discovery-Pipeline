"""Absorption predictions: Caco-2 permeability, HIA, Pgp substrate, oral bioavailability.

All predictions are rule-based using established medicinal chemistry thresholds
and RDKit molecular descriptors.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from drugflow.core.constants import (
    ADMET_BIOAVAILABILITY_LOGP_RANGE,
    ADMET_CACO2_TPSA_GOOD,
    ADMET_CACO2_TPSA_MODERATE,
    ADMET_HIA_TPSA_CUTOFF,
    ADMET_PGP_MW_THRESHOLD,
    ADMET_PGP_TPSA_THRESHOLD,
    ADMET_RISK_GREEN,
    ADMET_RISK_RED,
    ADMET_RISK_YELLOW,
    LIPINSKI_HBA_MAX,
    LIPINSKI_HBD_MAX,
    LIPINSKI_LOGP_MAX,
    LIPINSKI_MW_MAX,
    VEBER_ROTATABLE_BONDS_MAX,
    VEBER_TPSA_MAX,
)
from drugflow.core.exceptions import ADMETError
from drugflow.core.logging import get_logger
from drugflow.core.models import MoleculeRecord

logger = get_logger("admet.absorption")


@dataclass
class AbsorptionResult:
    """Container for absorption prediction results."""

    caco2_class: str           # "high", "moderate", "low"
    caco2_risk: str            # green/yellow/red
    hia_class: str             # "high", "moderate", "low"
    hia_risk: str
    pgp_substrate: bool        # True = likely substrate
    pgp_risk: str
    bioavailability_score: float  # 0-1
    bioavailability_risk: str
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to flat dictionary with admet_ prefix."""
        result = {
            "admet_caco2_class": self.caco2_class,
            "admet_caco2_risk": self.caco2_risk,
            "admet_hia_class": self.hia_class,
            "admet_hia_risk": self.hia_risk,
            "admet_pgp_substrate": self.pgp_substrate,
            "admet_pgp_risk": self.pgp_risk,
            "admet_bioavailability_score": self.bioavailability_score,
            "admet_bioavailability_risk": self.bioavailability_risk,
        }
        result.update(self.details)
        return result


def predict_caco2(mol: Chem.Mol) -> Tuple[str, str]:
    """Predict Caco-2 permeability class from TPSA.

    TPSA < 90 Å² → high permeability (green)
    TPSA 90-140 Å² → moderate permeability (yellow)
    TPSA > 140 Å² → low permeability (red)

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (class_str, risk_color)
    """
    tpsa = Descriptors.TPSA(mol)
    if tpsa < ADMET_CACO2_TPSA_GOOD:
        return "high", ADMET_RISK_GREEN
    elif tpsa < ADMET_CACO2_TPSA_MODERATE:
        return "moderate", ADMET_RISK_YELLOW
    else:
        return "low", ADMET_RISK_RED


def predict_hia(mol: Chem.Mol) -> Tuple[str, str]:
    """Predict Human Intestinal Absorption class.

    Combines Lipinski violations, Veber criteria, and TPSA.
    High HIA: ≤1 Lipinski violation AND TPSA < 140 AND RotBonds ≤ 10
    Moderate HIA: 2 Lipinski violations OR borderline TPSA
    Low HIA: 3+ Lipinski violations OR TPSA > 140

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (class_str, risk_color)
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)

    # Count Lipinski violations
    violations = 0
    if mw > LIPINSKI_MW_MAX:
        violations += 1
    if logp > LIPINSKI_LOGP_MAX:
        violations += 1
    if hbd > LIPINSKI_HBD_MAX:
        violations += 1
    if hba > LIPINSKI_HBA_MAX:
        violations += 1

    # Veber violations
    veber_fail = tpsa > VEBER_TPSA_MAX or rot_bonds > VEBER_ROTATABLE_BONDS_MAX

    if violations <= 1 and not veber_fail and tpsa < ADMET_HIA_TPSA_CUTOFF:
        return "high", ADMET_RISK_GREEN
    elif violations <= 2 and tpsa < ADMET_HIA_TPSA_CUTOFF:
        return "moderate", ADMET_RISK_YELLOW
    else:
        return "low", ADMET_RISK_RED


def predict_pgp_substrate(mol: Chem.Mol) -> Tuple[bool, str]:
    """Predict P-glycoprotein substrate likelihood.

    MW > 400 AND TPSA > 75 → likely Pgp substrate (efflux risk).

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (is_substrate, risk_color)
    """
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)

    is_substrate = mw > ADMET_PGP_MW_THRESHOLD and tpsa > ADMET_PGP_TPSA_THRESHOLD
    risk = ADMET_RISK_YELLOW if is_substrate else ADMET_RISK_GREEN
    return is_substrate, risk


def predict_oral_bioavailability(mol: Chem.Mol) -> Tuple[float, str]:
    """Compute oral bioavailability score (0-1).

    Weighted combination:
    - Lipinski compliance (0.3): 1.0 if ≤1 violation, 0.5 if 2, 0.0 if 3+
    - Veber compliance (0.3): 1.0 if passes, 0.0 if fails
    - LogP in drug-like range (0.2): 1.0 if in range, 0.0 if outside
    - TPSA penalty (0.2): 1.0 if <90, 0.5 if 90-140, 0.0 if >140

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (score, risk_color)
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)

    # Lipinski component
    violations = sum([
        mw > LIPINSKI_MW_MAX,
        logp > LIPINSKI_LOGP_MAX,
        hbd > LIPINSKI_HBD_MAX,
        hba > LIPINSKI_HBA_MAX,
    ])
    if violations <= 1:
        lipinski_score = 1.0
    elif violations == 2:
        lipinski_score = 0.5
    else:
        lipinski_score = 0.0

    # Veber component
    veber_score = 1.0 if (tpsa <= VEBER_TPSA_MAX and rot_bonds <= VEBER_ROTATABLE_BONDS_MAX) else 0.0

    # LogP range component
    logp_min, logp_max = ADMET_BIOAVAILABILITY_LOGP_RANGE
    logp_score = 1.0 if logp_min <= logp <= logp_max else 0.0

    # TPSA component
    if tpsa < ADMET_CACO2_TPSA_GOOD:
        tpsa_score = 1.0
    elif tpsa < ADMET_CACO2_TPSA_MODERATE:
        tpsa_score = 0.5
    else:
        tpsa_score = 0.0

    # Weighted combination
    score = (
        0.3 * lipinski_score
        + 0.3 * veber_score
        + 0.2 * logp_score
        + 0.2 * tpsa_score
    )

    if score >= 0.7:
        risk = ADMET_RISK_GREEN
    elif score >= 0.4:
        risk = ADMET_RISK_YELLOW
    else:
        risk = ADMET_RISK_RED

    return score, risk


def predict_absorption(mol: Chem.Mol) -> AbsorptionResult:
    """Run all absorption predictions for a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    AbsorptionResult
        Container with all absorption predictions.

    Raises
    ------
    ADMETError
        If molecule is None.
    """
    if mol is None:
        raise ADMETError("Cannot predict absorption for None molecule")

    caco2_cls, caco2_risk = predict_caco2(mol)
    hia_cls, hia_risk = predict_hia(mol)
    pgp_sub, pgp_risk = predict_pgp_substrate(mol)
    bioav_score, bioav_risk = predict_oral_bioavailability(mol)

    details = {
        "admet_tpsa": Descriptors.TPSA(mol),
        "admet_mw": Descriptors.MolWt(mol),
        "admet_logp": Descriptors.MolLogP(mol),
    }

    return AbsorptionResult(
        caco2_class=caco2_cls,
        caco2_risk=caco2_risk,
        hia_class=hia_cls,
        hia_risk=hia_risk,
        pgp_substrate=pgp_sub,
        pgp_risk=pgp_risk,
        bioavailability_score=bioav_score,
        bioavailability_risk=bioav_risk,
        details=details,
    )


def predict_absorption_record(rec: MoleculeRecord) -> None:
    """Predict absorption and store results in rec.properties.

    Parameters
    ----------
    rec : MoleculeRecord
        Molecule record with valid mol object.
    """
    if rec.mol is None:
        raise ADMETError(f"Cannot predict absorption for record {rec.record_id}: mol is None")

    result = predict_absorption(rec.mol)
    rec.properties.update(result.to_dict())
