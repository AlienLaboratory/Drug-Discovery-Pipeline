"""Lead Optimization pipeline: optimize a lead compound.

Takes a single lead molecule, generates analogs via mutations
and scaffold decoration, scores them, prepares 3D, and
optionally shape-screens against the original lead.
"""

import logging
import os
from typing import Any, Dict, Optional

from rdkit import Chem

from drugflow.core.exceptions import DockingError
from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.phase5.reporting.export import export_project_json, export_results_csv
from drugflow.phase5.reporting.summary import (
    create_pipeline_report,
    summarize_dataset,
    summarize_scoring,
)

logger = logging.getLogger(__name__)


def run_lead_optimization(
    lead_smiles: str,
    output_dir: str,
    n_analogs: int = 50,
    top_n: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run lead optimization pipeline.

    Phase 3: Generate analogs (mutations + scaffold decoration)
    Phase 1+2: Validate → Properties → Filters → Score
    Phase 4: 3D prep → Shape screen vs lead
    Rank by combined score (composite + shape similarity)
    Export: optimized candidates + report

    Args:
        lead_smiles: SMILES string of the lead molecule.
        output_dir: Output directory.
        n_analogs: Number of analogs to generate.
        top_n: Number of top candidates to keep.
        seed: Random seed.

    Returns:
        Pipeline report dict.
    """
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.data.writers import write_file
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from drugflow.phase1.analysis.filters import filter_dataset
    from drugflow.phase2.scoring.sa_score import compute_sa_score_dataset
    from drugflow.phase2.scoring.drug_likeness import compute_drug_likeness_dataset
    from drugflow.phase2.scoring.multi_objective import compute_composite_score_dataset
    from drugflow.phase2.scoring.ranking import rank_molecules
    from drugflow.phase3.generation.mutations import generate_mutations
    from drugflow.phase3.generation.scaffold_decoration import generate_from_scaffold

    os.makedirs(output_dir, exist_ok=True)
    stages = {}

    # Parse lead molecule
    lead_mol = Chem.MolFromSmiles(lead_smiles)
    if lead_mol is None:
        raise ValueError(f"Invalid lead SMILES: {lead_smiles}")

    lead_rec = MoleculeRecord(
        mol=lead_mol, smiles=lead_smiles,
        source_id="lead", status=MoleculeStatus.RAW,
    )

    stages["lead"] = {
        "smiles": lead_smiles,
        "n_atoms": lead_mol.GetNumHeavyAtoms(),
    }

    # ── Phase 3: Generate analogs ──
    logger.info("Phase 3: Generating analogs...")
    seed_dataset = MoleculeDataset(
        records=[lead_rec], name="lead_seed",
    )

    # Mutations
    n_mutations = n_analogs // 2
    mutants = generate_mutations(
        seed_dataset, n_molecules=max(n_mutations, 10),
        n_mutations_per_mol=1, seed=seed,
    )

    # Scaffold decoration
    n_decorated = n_analogs - n_mutations
    try:
        decorated = generate_from_scaffold(
            lead_mol, n_molecules=max(n_decorated, 10), seed=seed,
        )
    except Exception as e:
        logger.warning(f"Scaffold decoration failed: {e}")
        decorated = MoleculeDataset(records=[], name="decorated_empty")

    # Combine
    all_analogs = MoleculeDataset(
        records=mutants.records + decorated.records,
        name="analogs_combined",
    )

    stages["phase3_generation"] = {
        "n_mutations": len(mutants.valid_records),
        "n_decorated": len(decorated.valid_records),
        "n_total": len(all_analogs.records),
    }

    # ── Phase 1+2: Score ──
    logger.info("Phase 1+2: Validating and scoring analogs...")
    all_analogs = validate_dataset(all_analogs)
    all_analogs = compute_properties_dataset(all_analogs)
    all_analogs = compute_fingerprints_dataset(all_analogs)
    filter_dataset(all_analogs, lipinski=True, pains=True)
    all_analogs = compute_sa_score_dataset(all_analogs)
    all_analogs = compute_drug_likeness_dataset(all_analogs)
    all_analogs = compute_composite_score_dataset(all_analogs)

    stages["phase2_scoring"] = summarize_scoring(all_analogs)

    # ── Phase 4: 3D prep + Shape screen ──
    logger.info("Phase 4: 3D preparation and shape screening...")
    ranked = rank_molecules(all_analogs, "composite_score")
    top_candidates = [rec for _, rec, _ in ranked[:min(top_n, len(ranked))]]

    n_prepared = 0
    if top_candidates:
        from drugflow.phase4.ligand_prep.preparation import prepare_ligand

        # Prepare lead 3D
        try:
            lead_3d = prepare_ligand(lead_mol, n_confs=3, optimize=True, seed=seed)
        except Exception:
            lead_3d = None

        for rec in top_candidates:
            if rec.mol is not None:
                try:
                    rec.mol = prepare_ligand(rec.mol, n_confs=3, optimize=True, seed=seed)
                    n_prepared += 1

                    # Shape screen vs lead
                    if lead_3d is not None:
                        try:
                            from drugflow.phase4.shape_screening.scoring import compute_shape_tanimoto
                            score = compute_shape_tanimoto(rec.mol, lead_3d)
                            rec.properties["shape_similarity_to_lead"] = score
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"3D prep failed for {rec.source_id}: {e}")

    stages["phase4_3d_prep"] = {
        "n_candidates": len(top_candidates),
        "n_prepared_3d": n_prepared,
    }

    # ── Export ──
    logger.info("Exporting results...")
    result_dataset = MoleculeDataset(
        records=list(top_candidates) if top_candidates else [],
        name="lead_optimization_results",
    )

    export_results_csv(
        result_dataset,
        os.path.join(output_dir, "optimized_candidates.csv"),
        columns=[
            "MolWt", "LogP", "TPSA", "QED", "sa_score",
            "composite_score", "shape_similarity_to_lead",
            "lipinski_pass", "pains_pass",
        ],
    )
    write_file(all_analogs, os.path.join(output_dir, "all_analogs.csv"))

    report = create_pipeline_report(
        stages,
        metadata={
            "workflow": "lead_optimization",
            "lead_smiles": lead_smiles,
            "n_analogs": n_analogs,
            "top_n": top_n,
        },
    )
    export_project_json(report, os.path.join(output_dir, "report_summary.json"))

    logger.info(f"Lead optimization complete. Results in {output_dir}")
    return report
