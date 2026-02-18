"""Metabolism predictions: CYP inhibition risk and metabolic stability.

Uses SMARTS-based structural alert detection for CYP inhibition and
rule-based metabolic stability estimation from physicochemical properties.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from drugflow.core.constants import (
    ADMET_CYP_INHIBITION_SMARTS,
    ADMET_METABOLIC_STABILITY_AROMATIC_CUTOFF,
    ADMET_METABOLIC_STABILITY_LOGP_CUTOFF,
    ADMET_METABOLIC_STABILITY_MW_CUTOFF,
    ADMET_METABOLIC_STABILITY_ROTBONDS_CUTOFF,
    ADMET_RISK_GREEN,
    ADMET_RISK_RED,
    ADMET_RISK_YELLOW,
)
from drugflow.core.exceptions import ADMETError
from drugflow.core.logging import get_logger
from drugflow.core.models import MoleculeRecord

logger = get_logger("admet.metabolism")


@dataclass
class MetabolismResult:
    """Container for metabolism prediction results."""

    cyp_alerts: List[str]             # list of matched CYP alert names
    cyp_inhibition_risk: str          # "low", "moderate", "high"
    cyp_risk: str                     # green/yellow/red
    metabolic_stability: str          # "high", "moderate", "low"
    metabolic_stability_score: float  # 0-1 (1 = most stable)
    metabolic_risk: str               # green/yellow/red
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to flat dictionary with admet_ prefix."""
        return {
            "admet_cyp_alerts": ",".join(self.cyp_alerts) if self.cyp_alerts else "",
            "admet_cyp_n_alerts": len(self.cyp_alerts),
            "admet_cyp_inhibition_risk": self.cyp_inhibition_risk,
            "admet_cyp_risk": self.cyp_risk,
            "admet_metabolic_stability": self.metabolic_stability,
            "admet_metabolic_stability_score": self.metabolic_stability_score,
            "admet_metabolic_risk": self.metabolic_risk,
        }


def detect_cyp_alerts(mol: Chem.Mol) -> List[str]:
    """Detect CYP inhibition structural alerts via SMARTS matching.

    Matches against known CYP-inhibiting substructures: imidazole,
    triazole, pyridine, thiophene, furan, methylenedioxy, etc.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    list of str
        Names of matched CYP alert patterns.
    """
    alerts = []
    for name, smarts in ADMET_CYP_INHIBITION_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            alerts.append(name)
    return alerts


def predict_cyp_inhibition(mol: Chem.Mol) -> Tuple[str, List[str], str]:
    """Predict CYP inhibition risk level.

    Based on number of structural alert matches:
    0 alerts → low risk (green)
    1 alert → moderate risk (yellow)
    2+ alerts → high risk (red)

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (risk_class, alerts_list, risk_color)
    """
    alerts = detect_cyp_alerts(mol)
    n = len(alerts)

    if n == 0:
        return "low", alerts, ADMET_RISK_GREEN
    elif n == 1:
        return "moderate", alerts, ADMET_RISK_YELLOW
    else:
        return "high", alerts, ADMET_RISK_RED


def predict_metabolic_stability(mol: Chem.Mol) -> Tuple[str, float, str]:
    """Estimate metabolic stability from physicochemical properties.

    Penalty-based scoring where each risk factor reduces stability:
    - MW > 500 → penalty (larger molecules metabolized faster)
    - LogP > 3 → penalty (lipophilic = more CYP metabolism)
    - RotatableBonds > 7 → penalty (flexible = more metabolic sites)
    - AromaticRings > 3 → penalty (more aromatic rings = more CYP sites)

    Score 0-1 where 1 = high stability (fewest penalties).

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (stability_class, score_0_1, risk_color)
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    # Count penalties (each worth 0.25)
    penalties = 0
    if mw > ADMET_METABOLIC_STABILITY_MW_CUTOFF:
        penalties += 1
    if logp > ADMET_METABOLIC_STABILITY_LOGP_CUTOFF:
        penalties += 1
    if rot_bonds > ADMET_METABOLIC_STABILITY_ROTBONDS_CUTOFF:
        penalties += 1
    if aromatic_rings > ADMET_METABOLIC_STABILITY_AROMATIC_CUTOFF:
        penalties += 1

    score = max(0.0, 1.0 - penalties * 0.25)

    if score >= 0.75:
        cls = "high"
        risk = ADMET_RISK_GREEN
    elif score >= 0.5:
        cls = "moderate"
        risk = ADMET_RISK_YELLOW
    else:
        cls = "low"
        risk = ADMET_RISK_RED

    return cls, round(score, 3), risk


def predict_metabolism(mol: Chem.Mol) -> MetabolismResult:
    """Run all metabolism predictions for a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    MetabolismResult
        Container with all metabolism predictions.

    Raises
    ------
    ADMETError
        If molecule is None.
    """
    if mol is None:
        raise ADMETError("Cannot predict metabolism for None molecule")

    cyp_risk_cls, cyp_alerts, cyp_risk = predict_cyp_inhibition(mol)
    stab_cls, stab_score, stab_risk = predict_metabolic_stability(mol)

    return MetabolismResult(
        cyp_alerts=cyp_alerts,
        cyp_inhibition_risk=cyp_risk_cls,
        cyp_risk=cyp_risk,
        metabolic_stability=stab_cls,
        metabolic_stability_score=stab_score,
        metabolic_risk=stab_risk,
    )


def predict_metabolism_record(rec: MoleculeRecord) -> None:
    """Predict metabolism and store results in rec.properties.

    Parameters
    ----------
    rec : MoleculeRecord
        Molecule record with valid mol object.
    """
    if rec.mol is None:
        raise ADMETError(f"Cannot predict metabolism for record {rec.record_id}: mol is None")

    result = predict_metabolism(rec.mol)
    rec.properties.update(result.to_dict())
