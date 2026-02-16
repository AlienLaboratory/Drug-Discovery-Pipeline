"""Tests for molecule ranking."""

import pytest
from rdkit import Chem

from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.phase2.scoring.ranking import (
    rank_molecules,
    get_top_candidates,
    flag_candidates,
    export_ranked_results,
)


@pytest.fixture
def scored_dataset():
    """Dataset with composite scores and real molecules."""
    # Use short chain alkanes so all parse correctly
    smiles_list = [
        "C", "CC", "CCC", "CCCC", "CCCCC",
        "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC",
    ]
    records = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        rec = MoleculeRecord(
            mol=mol, source_id=f"mol_{i}",
            smiles=smi,
            status=MoleculeStatus.RAW,
        )
        rec.properties["composite_score"] = float(i) / 10.0
        rec.properties["drug_likeness_score"] = 0.5 + float(i) / 20.0
        rec.properties["sa_score"] = 2.0 + float(i) * 0.5
        records.append(rec)
    return MoleculeDataset(records=records, name="scored")


def test_rank_molecules(scored_dataset):
    """Ranking sorts by score descending."""
    ranked = rank_molecules(scored_dataset, "composite_score")
    assert len(ranked) == 10
    # First should have highest score
    assert ranked[0][2] >= ranked[-1][2]
    # Check rank property is set
    assert ranked[0][1].properties["rank"] == 1


def test_rank_molecules_ascending(scored_dataset):
    """Ascending ranking sorts low-to-high."""
    ranked = rank_molecules(
        scored_dataset, "composite_score", ascending=True,
    )
    assert ranked[0][2] <= ranked[-1][2]


def test_get_top_candidates(scored_dataset):
    """Get top-N candidates."""
    top = get_top_candidates(scored_dataset, top_n=3)
    assert len(top) == 3
    assert "top3" in top.name


def test_get_top_candidates_more_than_available(scored_dataset):
    """Top-N larger than dataset returns all."""
    top = get_top_candidates(scored_dataset, top_n=100)
    assert len(top) == 10


def test_flag_candidates(scored_dataset):
    """Flag candidates based on criteria."""
    flag_candidates(scored_dataset, criteria={
        "composite_score": (0.5, float("inf")),
    })
    flagged = sum(
        1 for r in scored_dataset.valid_records
        if r.properties.get("candidate_flag")
    )
    # Scores 0.5-0.9 = 5 molecules
    assert flagged == 5


def test_export_ranked_results(scored_dataset, tmp_path):
    """Export ranked results to CSV."""
    output = str(tmp_path / "ranked.csv")
    export_ranked_results(scored_dataset, output, top_n=5)

    import pandas as pd
    df = pd.read_csv(output)
    assert len(df) == 5
    assert "rank" in df.columns
    assert "composite_score" in df.columns
    assert df["rank"].tolist() == [1, 2, 3, 4, 5]


def test_export_ranked_results_all(scored_dataset, tmp_path):
    """Export all ranked results (no top_n)."""
    output = str(tmp_path / "all_ranked.csv")
    export_ranked_results(scored_dataset, output)

    import pandas as pd
    df = pd.read_csv(output)
    assert len(df) == 10
