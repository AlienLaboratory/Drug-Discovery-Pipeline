"""Toxicity predictions: hERG liability, AMES mutagenicity, hepatotoxicity, MRDD.

Uses SMARTS-based structural alert detection for toxicity endpoints
and rule-based risk assessment from physicochemical properties.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors

from drugflow.core.constants import (
    ADMET_AMES_SMARTS,
    ADMET_HEPATOTOX_SMARTS,
    ADMET_HERG_LOGP_THRESHOLD,
    ADMET_HERG_MW_THRESHOLD,
    ADMET_HERG_SMARTS,
    ADMET_RISK_GREEN,
    ADMET_RISK_RED,
    ADMET_RISK_YELLOW,
)
from drugflow.core.exceptions import ADMETError
from drugflow.core.logging import get_logger
from drugflow.core.models import MoleculeRecord

logger = get_logger("admet.toxicity")


@dataclass
class ToxicityResult:
    """Container for toxicity prediction results."""

    herg_risk_flag: bool           # True = risk
    herg_alerts: List[str]
    herg_risk: str                 # green/yellow/red
    ames_alerts: List[str]         # matched mutagenic alert names
    ames_risk: str
    hepatotox_alerts: List[str]    # reactive metabolite alerts
    hepatotox_risk: str
    mrdd_class: str                # "high_dose", "moderate_dose", "low_dose"
    mrdd_risk: str
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to flat dictionary with admet_ prefix."""
        return {
            "admet_herg_risk_flag": self.herg_risk_flag,
            "admet_herg_alerts": ",".join(self.herg_alerts) if self.herg_alerts else "",
            "admet_herg_n_alerts": len(self.herg_alerts),
            "admet_herg_risk": self.herg_risk,
            "admet_ames_alerts": ",".join(self.ames_alerts) if self.ames_alerts else "",
            "admet_ames_n_alerts": len(self.ames_alerts),
            "admet_ames_risk": self.ames_risk,
            "admet_hepatotox_alerts": ",".join(self.hepatotox_alerts) if self.hepatotox_alerts else "",
            "admet_hepatotox_n_alerts": len(self.hepatotox_alerts),
            "admet_hepatotox_risk": self.hepatotox_risk,
            "admet_mrdd_class": self.mrdd_class,
            "admet_mrdd_risk": self.mrdd_risk,
        }


def _match_smarts_patterns(mol: Chem.Mol, patterns: Dict[str, str]) -> List[str]:
    """Match a molecule against a dictionary of SMARTS patterns.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    patterns : dict
        Name → SMARTS mapping.

    Returns
    -------
    list of str
        Names of matched patterns.
    """
    alerts = []
    for name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            alerts.append(name)
    return alerts


def detect_herg_risk(mol: Chem.Mol) -> Tuple[bool, List[str], str]:
    """Predict hERG channel liability.

    hERG blockade causes QT prolongation → cardiac arrhythmia risk.
    Rule-based flags:
    - LogP > 3.7 AND MW > 400 → property-based risk
    - Structural alerts: basic nitrogen with long chain, phenothiazine, etc.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (has_risk, alerts_list, risk_color)
    """
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)

    # Property-based flag
    property_risk = logp > ADMET_HERG_LOGP_THRESHOLD and mw > ADMET_HERG_MW_THRESHOLD

    # Structural alerts
    alerts = _match_smarts_patterns(mol, ADMET_HERG_SMARTS)

    has_risk = property_risk or len(alerts) > 0

    if property_risk and len(alerts) > 0:
        risk = ADMET_RISK_RED
    elif has_risk:
        risk = ADMET_RISK_YELLOW
    else:
        risk = ADMET_RISK_GREEN

    if property_risk:
        alerts.append("high_logp_mw")

    return has_risk, alerts, risk


def detect_ames_alerts(mol: Chem.Mol) -> Tuple[List[str], str]:
    """Detect AMES mutagenicity structural alerts.

    Matches against known mutagenic substructures: aromatic nitro,
    aromatic amine, epoxides, aziridines, Michael acceptors,
    nitrogen mustards, etc.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (alerts_list, risk_color)
    """
    alerts = _match_smarts_patterns(mol, ADMET_AMES_SMARTS)

    if len(alerts) == 0:
        risk = ADMET_RISK_GREEN
    elif len(alerts) == 1:
        risk = ADMET_RISK_YELLOW
    else:
        risk = ADMET_RISK_RED

    return alerts, risk


def detect_hepatotox_alerts(mol: Chem.Mol) -> Tuple[List[str], str]:
    """Detect hepatotoxicity reactive metabolite alerts.

    Matches against substructures known to form reactive metabolites
    that can cause liver damage: quinones, thiophenes, furans,
    anilines, nitroaromatics, Michael acceptors, epoxides, etc.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (alerts_list, risk_color)
    """
    alerts = _match_smarts_patterns(mol, ADMET_HEPATOTOX_SMARTS)

    if len(alerts) == 0:
        risk = ADMET_RISK_GREEN
    elif len(alerts) <= 2:
        risk = ADMET_RISK_YELLOW
    else:
        risk = ADMET_RISK_RED

    return alerts, risk


def estimate_mrdd(mol: Chem.Mol) -> Tuple[str, str]:
    """Estimate maximum recommended daily dose class.

    Very potent compounds (low MW, high LogP, high complexity)
    typically require lower doses. Higher doses mean more
    potential for side effects.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    tuple of (mrdd_class, risk_color)
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)

    # Low MW + moderate LogP + moderate TPSA → likely potent → low dose OK
    # High MW + extreme LogP → likely needs high dose
    if mw < 400 and 1.0 <= logp <= 4.0 and tpsa < 120:
        return "low_dose", ADMET_RISK_GREEN
    elif mw > 600 or logp > 5.0 or logp < -1.0:
        return "high_dose", ADMET_RISK_YELLOW
    else:
        return "moderate_dose", ADMET_RISK_GREEN


def predict_toxicity(mol: Chem.Mol) -> ToxicityResult:
    """Run all toxicity predictions for a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    ToxicityResult
        Container with all toxicity predictions.

    Raises
    ------
    ADMETError
        If molecule is None.
    """
    if mol is None:
        raise ADMETError("Cannot predict toxicity for None molecule")

    herg_flag, herg_alerts, herg_risk = detect_herg_risk(mol)
    ames_alerts, ames_risk = detect_ames_alerts(mol)
    hepatotox_alerts, hepatotox_risk = detect_hepatotox_alerts(mol)
    mrdd_cls, mrdd_risk = estimate_mrdd(mol)

    return ToxicityResult(
        herg_risk_flag=herg_flag,
        herg_alerts=herg_alerts,
        herg_risk=herg_risk,
        ames_alerts=ames_alerts,
        ames_risk=ames_risk,
        hepatotox_alerts=hepatotox_alerts,
        hepatotox_risk=hepatotox_risk,
        mrdd_class=mrdd_cls,
        mrdd_risk=mrdd_risk,
    )


def predict_toxicity_record(rec: MoleculeRecord) -> None:
    """Predict toxicity and store results in rec.properties.

    Parameters
    ----------
    rec : MoleculeRecord
        Molecule record with valid mol object.
    """
    if rec.mol is None:
        raise ADMETError(f"Cannot predict toxicity for record {rec.record_id}: mol is None")

    result = predict_toxicity(rec.mol)
    rec.properties.update(result.to_dict())
