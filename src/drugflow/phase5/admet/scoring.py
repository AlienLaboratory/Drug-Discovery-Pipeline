"""Aggregate ADMET scoring and risk classification.

Combines individual domain scores (absorption, distribution, metabolism,
excretion, toxicity) into a composite ADMET score with traffic-light
risk classification.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from drugflow.core.constants import (
    ADMET_CLASS_FAVORABLE_THRESHOLD,
    ADMET_CLASS_MODERATE_THRESHOLD,
    ADMET_RISK_GREEN,
    ADMET_RISK_RED,
    ADMET_RISK_YELLOW,
    ADMET_SCORING_WEIGHTS,
)
from drugflow.core.exceptions import ADMETError
from drugflow.core.logging import get_logger
from drugflow.core.models import MoleculeRecord

logger = get_logger("admet.scoring")


@dataclass
class ADMETScore:
    """Aggregate ADMET score with risk breakdown."""

    overall_score: float           # 0-1 composite
    overall_class: str             # "favorable", "moderate", "poor"
    absorption_score: float        # 0-1 sub-score
    distribution_score: float
    metabolism_score: float
    excretion_score: float
    toxicity_score: float
    risk_summary: Dict[str, str] = field(default_factory=dict)
    n_red_flags: int = 0
    n_yellow_flags: int = 0
    n_green_flags: int = 0

    def to_dict(self) -> Dict:
        """Convert to flat dictionary with admet_ prefix."""
        return {
            "admet_score": self.overall_score,
            "admet_class": self.overall_class,
            "admet_absorption_score": self.absorption_score,
            "admet_distribution_score": self.distribution_score,
            "admet_metabolism_score": self.metabolism_score,
            "admet_excretion_score": self.excretion_score,
            "admet_toxicity_score": self.toxicity_score,
            "admet_n_red_flags": self.n_red_flags,
            "admet_n_yellow_flags": self.n_yellow_flags,
            "admet_n_green_flags": self.n_green_flags,
        }


def risk_to_score(risk: str) -> float:
    """Convert risk color to numeric score.

    Parameters
    ----------
    risk : str
        Risk color: "green", "yellow", or "red".

    Returns
    -------
    float
        1.0 for green, 0.5 for yellow, 0.0 for red.
    """
    if risk == ADMET_RISK_GREEN:
        return 1.0
    elif risk == ADMET_RISK_YELLOW:
        return 0.5
    elif risk == ADMET_RISK_RED:
        return 0.0
    else:
        return 0.5  # default for unknown


def classify_admet(score: float) -> str:
    """Classify overall ADMET profile.

    Parameters
    ----------
    score : float
        Aggregate ADMET score (0-1).

    Returns
    -------
    str
        "favorable" (>0.7), "moderate" (0.4-0.7), or "poor" (<0.4).
    """
    if score >= ADMET_CLASS_FAVORABLE_THRESHOLD:
        return "favorable"
    elif score >= ADMET_CLASS_MODERATE_THRESHOLD:
        return "moderate"
    else:
        return "poor"


def _collect_risk_flags(rec: MoleculeRecord) -> Dict[str, str]:
    """Collect all risk flags from rec.properties.

    Returns
    -------
    dict
        Property name → risk color mapping.
    """
    risk_keys = [
        "admet_caco2_risk",
        "admet_hia_risk",
        "admet_pgp_risk",
        "admet_bioavailability_risk",
        "admet_bbb_risk",
        "admet_ppb_risk",
        "admet_vod_risk",
        "admet_cyp_risk",
        "admet_metabolic_risk",
        "admet_renal_risk",
        "admet_halflife_risk",
        "admet_herg_risk",
        "admet_ames_risk",
        "admet_hepatotox_risk",
        "admet_mrdd_risk",
    ]
    return {k: rec.properties.get(k, ADMET_RISK_YELLOW) for k in risk_keys
            if k in rec.properties}


def compute_absorption_subscore(rec: MoleculeRecord) -> float:
    """Compute absorption sub-score (0-1) from individual risk flags.

    Averages: caco2, hia, pgp, bioavailability.
    For bioavailability, uses the score directly.
    """
    scores = []

    for key in ["admet_caco2_risk", "admet_hia_risk", "admet_pgp_risk"]:
        if key in rec.properties:
            scores.append(risk_to_score(rec.properties[key]))

    bioav = rec.properties.get("admet_bioavailability_score")
    if bioav is not None:
        scores.append(float(bioav))

    return sum(scores) / len(scores) if scores else 0.5


def compute_distribution_subscore(rec: MoleculeRecord) -> float:
    """Compute distribution sub-score (0-1) from individual risk flags."""
    scores = []
    for key in ["admet_bbb_risk", "admet_ppb_risk", "admet_vod_risk"]:
        if key in rec.properties:
            scores.append(risk_to_score(rec.properties[key]))
    return sum(scores) / len(scores) if scores else 0.5


def compute_metabolism_subscore(rec: MoleculeRecord) -> float:
    """Compute metabolism sub-score (0-1) from risk flags and stability score."""
    scores = []

    if "admet_cyp_risk" in rec.properties:
        scores.append(risk_to_score(rec.properties["admet_cyp_risk"]))

    stab = rec.properties.get("admet_metabolic_stability_score")
    if stab is not None:
        scores.append(float(stab))

    return sum(scores) / len(scores) if scores else 0.5


def compute_excretion_subscore(rec: MoleculeRecord) -> float:
    """Compute excretion sub-score (0-1) from risk flags."""
    scores = []
    for key in ["admet_renal_risk", "admet_halflife_risk"]:
        if key in rec.properties:
            scores.append(risk_to_score(rec.properties[key]))
    return sum(scores) / len(scores) if scores else 0.5


def compute_toxicity_subscore(rec: MoleculeRecord) -> float:
    """Compute toxicity sub-score (0-1). Heavily weighted toward red flags."""
    scores = []
    for key in ["admet_herg_risk", "admet_ames_risk",
                "admet_hepatotox_risk", "admet_mrdd_risk"]:
        if key in rec.properties:
            scores.append(risk_to_score(rec.properties[key]))
    return sum(scores) / len(scores) if scores else 0.5


def compute_admet_score(
    rec: MoleculeRecord,
    weights: Optional[Dict[str, float]] = None,
) -> ADMETScore:
    """Compute aggregate ADMET score for a molecule.

    Requires individual ADMET properties already stored in rec.properties
    (run predict_absorption_record, etc. first).

    Parameters
    ----------
    rec : MoleculeRecord
        Record with ADMET properties.
    weights : dict, optional
        Custom weights for domain contributions.
        Keys: "absorption", "distribution", "metabolism", "excretion", "toxicity".

    Returns
    -------
    ADMETScore
        Aggregate score with breakdown.
    """
    if weights is None:
        weights = ADMET_SCORING_WEIGHTS

    abs_score = compute_absorption_subscore(rec)
    dist_score = compute_distribution_subscore(rec)
    met_score = compute_metabolism_subscore(rec)
    exc_score = compute_excretion_subscore(rec)
    tox_score = compute_toxicity_subscore(rec)

    # Weighted combination
    total_weight = sum(weights.values())
    overall = (
        weights.get("absorption", 0.25) * abs_score
        + weights.get("distribution", 0.15) * dist_score
        + weights.get("metabolism", 0.20) * met_score
        + weights.get("excretion", 0.10) * exc_score
        + weights.get("toxicity", 0.30) * tox_score
    ) / total_weight

    overall = round(overall, 3)
    overall_class = classify_admet(overall)

    # Collect risk flags
    risk_summary = _collect_risk_flags(rec)
    n_red = sum(1 for v in risk_summary.values() if v == ADMET_RISK_RED)
    n_yellow = sum(1 for v in risk_summary.values() if v == ADMET_RISK_YELLOW)
    n_green = sum(1 for v in risk_summary.values() if v == ADMET_RISK_GREEN)

    return ADMETScore(
        overall_score=overall,
        overall_class=overall_class,
        absorption_score=round(abs_score, 3),
        distribution_score=round(dist_score, 3),
        metabolism_score=round(met_score, 3),
        excretion_score=round(exc_score, 3),
        toxicity_score=round(tox_score, 3),
        risk_summary=risk_summary,
        n_red_flags=n_red,
        n_yellow_flags=n_yellow,
        n_green_flags=n_green,
    )


def compute_admet_score_record(
    rec: MoleculeRecord,
    weights: Optional[Dict[str, float]] = None,
) -> None:
    """Compute ADMET score and store in rec.properties.

    Parameters
    ----------
    rec : MoleculeRecord
        Record with individual ADMET predictions already computed.
    weights : dict, optional
        Custom domain weights.
    """
    score = compute_admet_score(rec, weights)
    rec.properties.update(score.to_dict())
