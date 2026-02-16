"""De Novo Design pipeline: generate and score novel molecules.

Connects Phases 1, 3, 2, and 4: load seeds → generate →
validate → score → 3D prep → export.
"""

import logging
import os
from typing import Any, Dict, Optional

from drugflow.core.models import MoleculeDataset
from drugflow.phase5.reporting.export import export_project_json, export_results_csv
from drugflow.phase5.reporting.summary import (
    create_pipeline_report,
    summarize_dataset,
    summarize_generation,
    summarize_scoring,
)

logger = logging.getLogger(__name__)


def run_denovo_design(
    input_path: str,
    output_dir: str,
    strategy: str = "brics",
    model_path: Optional[str] = None,
    n_generate: int = 100,
    top_n: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run de novo molecular design pipeline.

    Phase 1: Load seeds → Validate → Properties → Fingerprints
    Phase 3: Generate novel molecules (BRICS, mutation, or GA)
    Phase 1+2: Validate generated → Properties → Filters → Score
    Phase 4: 3D prep for top candidates
    Export: scored results + report

    Args:
        input_path: Path to seed molecules file.
        output_dir: Output directory.
        strategy: Generation strategy ("brics", "mutate").
        model_path: Optional QSAR model for GA or prediction.
        n_generate: Number of molecules to generate.
        top_n: Number of top candidates to keep.
        seed: Random seed.

    Returns:
        Pipeline report dict.
    """
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.data.writers import write_file
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from drugflow.phase1.analysis.filters import filter_dataset
    from drugflow.phase2.scoring.sa_score import compute_sa_score_dataset
    from drugflow.phase2.scoring.drug_likeness import compute_drug_likeness_dataset
    from drugflow.phase2.scoring.multi_objective import compute_composite_score_dataset
    from drugflow.phase2.scoring.ranking import rank_molecules

    os.makedirs(output_dir, exist_ok=True)
    stages = {}

    # ── Phase 1: Load seeds ──
    logger.info("Phase 1: Loading seed molecules...")
    seed_dataset = load_file(path=input_path)
    seed_dataset = validate_dataset(seed_dataset)
    seed_dataset = compute_properties_dataset(seed_dataset)
    seed_dataset = compute_fingerprints_dataset(seed_dataset)

    stages["phase1_seeds"] = summarize_dataset(seed_dataset)

    # ── Phase 3: Generate ──
    logger.info(f"Phase 3: Generating molecules (strategy={strategy})...")

    if strategy == "brics":
        from drugflow.phase3.generation.brics_enum import generate_brics
        generated = generate_brics(seed_dataset, n_molecules=n_generate, seed=seed)
    elif strategy == "mutate":
        from drugflow.phase3.generation.mutations import generate_mutations
        generated = generate_mutations(
            seed_dataset, n_molecules=n_generate,
            n_mutations_per_mol=1, seed=seed,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'brics' or 'mutate'.")

    stages["phase3_generation"] = summarize_generation(seed_dataset, generated)

    # ── Phase 1+2: Score generated ──
    logger.info("Phase 1+2: Validating and scoring generated molecules...")
    generated = validate_dataset(generated)
    generated = compute_properties_dataset(generated)
    generated = compute_fingerprints_dataset(generated)
    filter_dataset(generated, lipinski=True, pains=True)

    # Optional QSAR prediction
    if model_path:
        from drugflow.phase1.analysis.descriptors import compute_descriptors_dataset
        from drugflow.phase2.qsar.persistence import load_model
        from drugflow.phase2.qsar.prediction import predict_dataset

        qsar_model = load_model(model_path)
        generated = compute_descriptors_dataset(generated)
        predict_dataset(qsar_model, generated, feature_source="descriptors")

    generated = compute_sa_score_dataset(generated)
    generated = compute_drug_likeness_dataset(generated)
    generated = compute_composite_score_dataset(generated)

    stages["phase2_scoring"] = summarize_scoring(generated)

    # ── Phase 4: 3D prep top candidates ──
    logger.info("Phase 4: 3D preparation for top candidates...")
    ranked = rank_molecules(generated, "composite_score")
    top_candidates = [rec for _, rec, _ in ranked[:min(top_n, len(ranked))]]

    if top_candidates:
        from drugflow.phase4.ligand_prep.preparation import prepare_ligand

        n_prepared = 0
        for rec in top_candidates:
            if rec.mol is not None:
                try:
                    rec.mol = prepare_ligand(rec.mol, n_confs=3, optimize=True, seed=seed)
                    n_prepared += 1
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
        name="denovo_results",
    )

    export_results_csv(result_dataset, os.path.join(output_dir, "generated_scored.csv"))
    write_file(generated, os.path.join(output_dir, "all_generated.csv"))

    report = create_pipeline_report(
        stages,
        metadata={
            "workflow": "denovo_design",
            "input": input_path,
            "strategy": strategy,
            "model": model_path,
            "n_generate": n_generate,
            "top_n": top_n,
        },
    )
    export_project_json(report, os.path.join(output_dir, "report_summary.json"))

    logger.info(f"De novo design complete. Results in {output_dir}")
    return report
