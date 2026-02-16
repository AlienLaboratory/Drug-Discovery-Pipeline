"""Shape-based virtual screening workflow.

Screens a library of molecules against a reference using 3D shape
similarity, with alignment and scoring.
"""

import logging
from typing import List, Optional, Tuple

from rdkit import Chem

from claudedd.core.constants import SHAPE_DEFAULT_METRIC, SHAPE_DEFAULT_THRESHOLD
from claudedd.core.exceptions import DockingError
from claudedd.core.models import MoleculeDataset, MoleculeRecord
from claudedd.phase4.shape_screening.alignment import align_molecules_o3a
from claudedd.phase4.shape_screening.scoring import (
    compute_combo_score,
    compute_shape_tanimoto,
)

logger = logging.getLogger(__name__)


def screen_by_shape(
    library: MoleculeDataset,
    reference: Chem.Mol,
    threshold: float = SHAPE_DEFAULT_THRESHOLD,
    metric: str = SHAPE_DEFAULT_METRIC,
    conf_id_ref: int = 0,
) -> MoleculeDataset:
    """Screen library molecules by shape similarity to reference.

    Args:
        library: Library of molecules to screen.
        reference: Reference molecule (must have 3D coords).
        threshold: Minimum shape score to pass.
        metric: "tanimoto" or "combo".
        conf_id_ref: Reference conformer ID.

    Returns:
        MoleculeDataset of hits passing threshold.
    """
    if reference is None:
        raise DockingError("Reference molecule cannot be None")

    hits = []
    for rec in library.valid_records:
        if rec.mol is None:
            continue
        try:
            # Align and score
            if metric == "combo":
                score = compute_combo_score(rec.mol, reference, conf_id2=conf_id_ref)
            else:
                # Align first for better shape score
                try:
                    _, aligned = align_molecules_o3a(rec.mol, reference, conf_id_ref=conf_id_ref)
                    rec.mol = aligned
                except DockingError:
                    pass
                score = compute_shape_tanimoto(rec.mol, reference, conf_id2=conf_id_ref)

            rec.properties[f"shape_{metric}"] = score
            rec.properties["shape_screen_pass"] = score >= threshold

            if score >= threshold:
                hits.append(rec)
        except DockingError as e:
            logger.debug(f"Shape screening failed for {rec.source_id}: {e}")
            continue

    result = MoleculeDataset(
        records=hits,
        name=f"shape_hits_{len(hits)}",
    )
    logger.info(
        f"Shape screening: {len(hits)} hits / {len(library.valid_records)} "
        f"(threshold={threshold}, metric={metric})"
    )
    return result


def rank_by_shape(
    library: MoleculeDataset,
    reference: Chem.Mol,
    metric: str = SHAPE_DEFAULT_METRIC,
    conf_id_ref: int = 0,
) -> List[Tuple[int, MoleculeRecord, float]]:
    """Rank library molecules by shape similarity to reference.

    Args:
        library: Library of molecules.
        reference: Reference molecule.
        metric: Score metric.
        conf_id_ref: Reference conformer ID.

    Returns:
        List of (index, record, score) tuples, sorted descending.
    """
    if reference is None:
        raise DockingError("Reference molecule cannot be None")

    scored = []
    for i, rec in enumerate(library.valid_records):
        if rec.mol is None:
            continue
        try:
            if metric == "combo":
                score = compute_combo_score(rec.mol, reference, conf_id2=conf_id_ref)
            else:
                score = compute_shape_tanimoto(rec.mol, reference, conf_id2=conf_id_ref)

            rec.properties[f"shape_{metric}"] = score
            scored.append((i, rec, score))
        except DockingError:
            continue

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored
