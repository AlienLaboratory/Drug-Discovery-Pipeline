"""Distribution predictions: BBB penetration, plasma protein binding, volume of distribution.

All predictions are rule-based using established pharmacokinetic models
and RDKit molecular descriptors.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors

from drugflow.core.constants import (
    ADMET_BBB_INTERCEPT,
    ADMET_BBB_LOGP_COEFF,
    ADMET_BBB_THRESHOLD,
    ADMET_BBB_TPSA_COEFF,
    ADMET_PPB_LOGP_HIGH,
    ADMET_PPB_LOGP_LOW,
    ADMET_RISK_GREEN,
    ADMET_RISK_RED,
    ADMET_RISK_YELLOW,
    ADMET_VOD_LOGP_HIGH,
    ADMET_VOD_LOGP_LOW,
)
from drugflow.core.exceptions import ADMETError
from drugflow.core.logging import get_logger
from drugflow.core.models import MoleculeRecord

logger = get_logger("admet.distribution")


@dataclass
class DistributionResult:
    """Container for distribution prediction results."""

    bbb_penetrant: bool        # True = likely BBB+
    bbb_logbb: float           # estimated LogBB value
    bbb_risk: str              # green/yellow/red
    ppb_class: str             # "high", "moderate", "low"
    ppb_risk: str
    vod_class: str             # "low", "moderate", "high"
    vod_risk: str
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to flat dictionary with admet_ prefix."""
        result = {
            "admet_bbb_penetrant": self.bbb_penetrant,
            "admet_bbb_logbb": self.bbb_logbb,
            "admet_bbb_risk": self.bbb_risk,
            "admet_ppb_class": self.ppb_class,
            "admet_ppb_risk": self.ppb_risk,
            "admet_vod_class": self.vod_class,
            "admet_vod_risk": self.vod_risk,
        }
        result.update(self.details)
        return result


def predict_bbb(mol: Chem.Mol) -> Tuple[bool, float, str]:
    """Predict Blood-Brain Barrier penetration.

    Uses the Clark equation: LogBB = 0.155*LogP - 0.01*TPSA + 0.164
    BBB+ if LogBB > 0.

    For drug discovery targeting peripheral targets (like BCL-2),
    BBB penetration may actually be undesirable (yellow risk if penetrant).

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (is_penetrant, logbb_value, risk_color)
    """
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)

    logbb = ADMET_BBB_LOGP_COEFF * logp + ADMET_BBB_TPSA_COEFF * tpsa + ADMET_BBB_INTERCEPT
    is_penetrant = logbb > ADMET_BBB_THRESHOLD

    # For most targets, BBB penetration is informational
    # Green = clear result (either way), yellow = borderline
    if abs(logbb) > 0.3:
        risk = ADMET_RISK_GREEN
    else:
        risk = ADMET_RISK_YELLOW

    return is_penetrant, round(logbb, 3), risk


def predict_ppb(mol: Chem.Mol) -> Tuple[str, str]:
    """Predict plasma protein binding class.

    LogP > 3 → high PPB (reduces free drug fraction → yellow risk)
    LogP 1-3 → moderate PPB
    LogP < 1 → low PPB (more free drug → green)

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (class_str, risk_color)
    """
    logp = Descriptors.MolLogP(mol)

    if logp > ADMET_PPB_LOGP_HIGH:
        return "high", ADMET_RISK_YELLOW
    elif logp > ADMET_PPB_LOGP_LOW:
        return "moderate", ADMET_RISK_GREEN
    else:
        return "low", ADMET_RISK_GREEN


def predict_vod(mol: Chem.Mol) -> Tuple[str, str]:
    """Estimate volume of distribution class.

    Low LogP → low VoD (stays in blood)
    Moderate LogP → moderate VoD (distributes to tissues)
    High LogP → high VoD (accumulates in fatty tissues)

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (class_str, risk_color)
    """
    logp = Descriptors.MolLogP(mol)

    if logp < ADMET_VOD_LOGP_LOW:
        return "low", ADMET_RISK_GREEN
    elif logp < ADMET_VOD_LOGP_HIGH:
        return "moderate", ADMET_RISK_GREEN
    else:
        return "high", ADMET_RISK_YELLOW


def predict_distribution(mol: Chem.Mol) -> DistributionResult:
    """Run all distribution predictions for a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    DistributionResult
        Container with all distribution predictions.

    Raises
    ------
    ADMETError
        If molecule is None.
    """
    if mol is None:
        raise ADMETError("Cannot predict distribution for None molecule")

    bbb_pen, bbb_logbb, bbb_risk = predict_bbb(mol)
    ppb_cls, ppb_risk = predict_ppb(mol)
    vod_cls, vod_risk = predict_vod(mol)

    return DistributionResult(
        bbb_penetrant=bbb_pen,
        bbb_logbb=bbb_logbb,
        bbb_risk=bbb_risk,
        ppb_class=ppb_cls,
        ppb_risk=ppb_risk,
        vod_class=vod_cls,
        vod_risk=vod_risk,
    )


def predict_distribution_record(rec: MoleculeRecord) -> None:
    """Predict distribution and store results in rec.properties.

    Parameters
    ----------
    rec : MoleculeRecord
        Molecule record with valid mol object.
    """
    if rec.mol is None:
        raise ADMETError(f"Cannot predict distribution for record {rec.record_id}: mol is None")

    result = predict_distribution(rec.mol)
    rec.properties.update(result.to_dict())
