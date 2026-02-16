"""Substructure and SMARTS-based virtual screening.

Finds molecules containing specified substructures using
SMARTS pattern matching or direct SMILES substructure search.
"""

from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from drugflow.core.exceptions import ScreeningError
from drugflow.core.logging import get_logger, progress_bar
from drugflow.core.models import MoleculeDataset, MoleculeRecord

logger = get_logger("screening.substructure")


def parse_pattern(pattern: str) -> Chem.Mol:
    """Parse a SMARTS or SMILES pattern string into a query molecule.

    Tries SMARTS first, falls back to SMILES.

    Parameters
    ----------
    pattern : str
        SMARTS or SMILES pattern string.

    Returns
    -------
    Chem.Mol
        Query molecule for substructure matching.

    Raises
    ------
    ScreeningError
        If pattern cannot be parsed.
    """
    # Try SMARTS first
    query = Chem.MolFromSmarts(pattern)
    if query is not None:
        return query

    # Fallback to SMILES
    query = Chem.MolFromSmiles(pattern)
    if query is not None:
        return query

    raise ScreeningError(f"Cannot parse pattern as SMARTS or SMILES: '{pattern}'")


def has_substructure(mol: Chem.Mol, pattern: Chem.Mol) -> bool:
    """Check if a molecule contains a substructure.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to check.
    pattern : Chem.Mol
        Query pattern (from SMARTS or SMILES).

    Returns
    -------
    bool
        True if molecule contains the substructure.
    """
    if mol is None or pattern is None:
        return False
    return mol.HasSubstructMatch(pattern)


def count_substructure_matches(mol: Chem.Mol, pattern: Chem.Mol) -> int:
    """Count non-overlapping substructure matches.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to check.
    pattern : Chem.Mol
        Query pattern.

    Returns
    -------
    int
        Number of non-overlapping matches.
    """
    if mol is None or pattern is None:
        return 0
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    return len(matches)


def get_substructure_matches(
    mol: Chem.Mol, pattern: Chem.Mol,
) -> List[Tuple[int, ...]]:
    """Get atom indices of all substructure matches.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to check.
    pattern : Chem.Mol
        Query pattern.

    Returns
    -------
    list of tuple
        Each tuple contains atom indices for one match.
    """
    if mol is None or pattern is None:
        return []
    return list(mol.GetSubstructMatches(pattern, uniquify=True))


def screen_substructure(
    dataset: MoleculeDataset,
    pattern: str,
    exclude: bool = False,
    count_matches: bool = False,
) -> MoleculeDataset:
    """Screen dataset for molecules matching a substructure pattern.

    Stores results in rec.properties:
      - "substruct_match": True/False
      - "substruct_pattern": the pattern used
      - "substruct_match_count": count of matches (if count_matches=True)

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset.
    pattern : str
        SMARTS or SMILES pattern to match.
    exclude : bool
        If True, marks molecules that do NOT contain the pattern as hits.
    count_matches : bool
        If True, also counts the number of matches per molecule.

    Returns
    -------
    MoleculeDataset
        New dataset containing only the hits.

    Raises
    ------
    ScreeningError
        If the pattern cannot be parsed.
    """
    query = parse_pattern(pattern)

    hits = []
    miss_count = 0

    for rec in progress_bar(dataset.valid_records, desc="Substructure screening"):
        if rec.mol is None:
            continue

        matched = has_substructure(rec.mol, query)

        # XOR with exclude flag
        is_hit = matched != exclude

        rec.properties["substruct_match"] = matched
        rec.properties["substruct_pattern"] = pattern

        if count_matches:
            n_matches = count_substructure_matches(rec.mol, query)
            rec.properties["substruct_match_count"] = n_matches

        rec.add_provenance(
            f"screen:substructure:{'hit' if is_hit else 'miss'}"
        )

        if is_hit:
            hits.append(rec)
        else:
            miss_count += 1

    logger.info(
        f"Substructure screen: {len(hits)} hits, {miss_count} misses "
        f"(pattern='{pattern}', exclude={exclude})"
    )

    result = MoleculeDataset(records=hits, name=f"{dataset.name}_substruct_hits")
    result._provenance = dataset._provenance + [f"screen:substructure:{pattern}"]
    return result


def screen_multi_substructure(
    dataset: MoleculeDataset,
    patterns: List[str],
    match_all: bool = False,
) -> MoleculeDataset:
    """Screen dataset against multiple substructure patterns.

    Parameters
    ----------
    dataset : MoleculeDataset
        Input dataset.
    patterns : list of str
        SMARTS or SMILES patterns.
    match_all : bool
        If True, molecule must match ALL patterns. If False, any match suffices.

    Returns
    -------
    MoleculeDataset
        New dataset containing hits.
    """
    queries = []
    for p in patterns:
        queries.append(parse_pattern(p))

    hits = []
    for rec in progress_bar(dataset.valid_records, desc="Multi-pattern screening"):
        if rec.mol is None:
            continue

        matches = [has_substructure(rec.mol, q) for q in queries]
        matched_patterns = [p for p, m in zip(patterns, matches) if m]

        if match_all:
            is_hit = all(matches)
        else:
            is_hit = any(matches)

        rec.properties["substruct_multi_match"] = is_hit
        rec.properties["substruct_matched_patterns"] = matched_patterns
        rec.properties["substruct_match_ratio"] = (
            sum(matches) / len(matches) if matches else 0.0
        )

        if is_hit:
            hits.append(rec)

    mode = "ALL" if match_all else "ANY"
    logger.info(
        f"Multi-substructure screen ({mode}): {len(hits)} hits "
        f"from {len(dataset.valid_records)} molecules, {len(patterns)} patterns"
    )

    result = MoleculeDataset(records=hits, name=f"{dataset.name}_multi_substruct_hits")
    return result
