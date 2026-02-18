"""Excretion predictions: renal clearance likelihood and half-life estimation.

Rule-based predictions using physicochemical properties to estimate
how a drug candidate is likely to be eliminated from the body.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors

from drugflow.core.constants import (
    ADMET_HALFLIFE_LONG,
    ADMET_HALFLIFE_MEDIUM,
    ADMET_HALFLIFE_SHORT,
    ADMET_RENAL_LOGP_THRESHOLD,
    ADMET_RENAL_MW_THRESHOLD,
    ADMET_RISK_GREEN,
    ADMET_RISK_RED,
    ADMET_RISK_YELLOW,
)
from drugflow.core.exceptions import ADMETError
from drugflow.core.logging import get_logger
from drugflow.core.models import MoleculeRecord

logger = get_logger("admet.excretion")


@dataclass
class ExcretionResult:
    """Container for excretion prediction results."""

    renal_clearance_likely: bool  # True = likely renally cleared
    renal_risk: str               # green/yellow/red
    halflife_class: str           # "short", "medium", "long"
    halflife_risk: str
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to flat dictionary with admet_ prefix."""
        result = {
            "admet_renal_clearance": self.renal_clearance_likely,
            "admet_renal_risk": self.renal_risk,
            "admet_halflife_class": self.halflife_class,
            "admet_halflife_risk": self.halflife_risk,
        }
        result.update(self.details)
        return result


def predict_renal_clearance(mol: Chem.Mol) -> Tuple[bool, str]:
    """Predict renal clearance likelihood.

    Small, hydrophilic molecules (MW < 500, LogP < 2) are more likely
    to be cleared renally. Renal clearance is generally favorable
    as it means predictable elimination.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (is_renal, risk_color)
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    is_renal = mw < ADMET_RENAL_MW_THRESHOLD and logp < ADMET_RENAL_LOGP_THRESHOLD
    # Renal clearance is informational — neither inherently good nor bad
    risk = ADMET_RISK_GREEN if is_renal else ADMET_RISK_YELLOW
    return is_renal, risk


def predict_halflife(
    mol: Chem.Mol,
    metabolic_stability_score: float = 0.5,
) -> Tuple[str, str]:
    """Estimate half-life class based on metabolic stability and MW.

    Combines metabolic stability score with molecular weight:
    - High stability + low MW → short half-life (quick clearance)
    - Low stability + high MW → long half-life (accumulation risk)
    - Moderate combinations → medium half-life

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    metabolic_stability_score : float
        Score from predict_metabolic_stability (0-1, 1=stable).

    Returns
    -------
    tuple of (halflife_class, risk_color)
    """
    mw = Descriptors.MolWt(mol)

    # High metabolic stability + small molecule → cleared quickly
    # Low metabolic stability + large molecule → accumulates
    if metabolic_stability_score >= 0.75 and mw < 400:
        cls = ADMET_HALFLIFE_SHORT
        risk = ADMET_RISK_GREEN
    elif metabolic_stability_score < 0.5 and mw > 500:
        cls = ADMET_HALFLIFE_LONG
        risk = ADMET_RISK_YELLOW
    else:
        cls = ADMET_HALFLIFE_MEDIUM
        risk = ADMET_RISK_GREEN

    return cls, risk


def predict_excretion(
    mol: Chem.Mol,
    metabolic_stability_score: float = 0.5,
) -> ExcretionResult:
    """Run all excretion predictions for a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    metabolic_stability_score : float
        Metabolic stability score from metabolism module (0-1).

    Returns
    -------
    ExcretionResult
        Container with all excretion predictions.

    Raises
    ------
    ADMETError
        If molecule is None.
    """
    if mol is None:
        raise ADMETError("Cannot predict excretion for None molecule")

    renal, renal_risk = predict_renal_clearance(mol)
    hl_cls, hl_risk = predict_halflife(mol, metabolic_stability_score)

    return ExcretionResult(
        renal_clearance_likely=renal,
        renal_risk=renal_risk,
        halflife_class=hl_cls,
        halflife_risk=hl_risk,
    )


def predict_excretion_record(rec: MoleculeRecord) -> None:
    """Predict excretion and store results in rec.properties.

    Uses metabolic_stability_score from metabolism module if available.

    Parameters
    ----------
    rec : MoleculeRecord
        Molecule record with valid mol object.
    """
    if rec.mol is None:
        raise ADMETError(f"Cannot predict excretion for record {rec.record_id}: mol is None")

    # Use metabolic stability score if already computed
    met_score = rec.properties.get("admet_metabolic_stability_score", 0.5)

    result = predict_excretion(rec.mol, metabolic_stability_score=met_score)
    rec.properties.update(result.to_dict())
