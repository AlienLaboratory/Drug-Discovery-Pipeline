"""Hit-to-Lead pipeline: end-to-end drug discovery workflow.

Connects all 4 phases: load → profile → screen → score →
generate analogs → rescore → 3D prepare → rank → export.
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


def run_hit_to_lead(
    input_path: str,
    output_dir: str,
    model_path: Optional[str] = None,
    n_generate: int = 50,
    top_n: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run complete hit-to-lead pipeline.

    Phase 1: Load → Validate → Standardize → Properties → Fingerprints → Filters
    Phase 2: (Optional QSAR) → SA Score → Drug-likeness → Composite Score
    Phase 3: Generate analogs from top hits → Rescore
    Phase 4: Prepare 3D for top candidates
    Export: ranked results + report

    Args:
        input_path: Path to input molecule file.
        output_dir: Output directory.
        model_path: Optional path to trained QSAR model.
        n_generate: Number of analogs to generate.
        top_n: Number of top candidates to keep.
        seed: Random seed.

    Returns:
        Pipeline report dict with results and summary.
    """
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.data.standardizer import standardize_dataset
    from drugflow.phase1.data.writers import write_file
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase1.analysis.descriptors import compute_descriptors_dataset
    from drugflow.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from drugflow.phase1.analysis.filters import filter_dataset
    from drugflow.phase2.scoring.sa_score import compute_sa_score_dataset
    from drugflow.phase2.scoring.drug_likeness import compute_drug_likeness_dataset
    from drugflow.phase2.scoring.multi_objective import compute_composite_score_dataset
    from drugflow.phase2.scoring.ranking import rank_molecules

    os.makedirs(output_dir, exist_ok=True)
    stages = {}

    # ── Phase 1: Load & Profile ──
    logger.info("Phase 1: Loading and profiling...")
    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)
    dataset = standardize_dataset(dataset)
    dataset = compute_properties_dataset(dataset)
    dataset = compute_fingerprints_dataset(dataset)
    filter_dataset(dataset, lipinski=True, pains=True)

    stages["phase1_profile"] = summarize_dataset(dataset)

    # ── Phase 2: Score ──
    logger.info("Phase 2: Scoring...")
    if model_path:
        from drugflow.phase2.qsar.persistence import load_model
        from drugflow.phase2.qsar.prediction import predict_dataset

        qsar_model = load_model(model_path)
        dataset = compute_descriptors_dataset(dataset)
        predict_dataset(qsar_model, dataset, feature_source="descriptors")

    dataset = compute_sa_score_dataset(dataset)
    dataset = compute_drug_likeness_dataset(dataset)
    dataset = compute_composite_score_dataset(dataset)

    stages["phase2_scoring"] = summarize_scoring(dataset)

    # ── Phase 3: Generate analogs from top hits ──
    logger.info("Phase 3: Generating analogs...")
    ranked = rank_molecules(dataset, "composite_score")
    top_hits = [rec for _, rec, _ in ranked[:min(10, len(ranked))]]

    if top_hits:
        from drugflow.core.models import MoleculeRecord, MoleculeStatus
        from drugflow.phase3.generation.mutations import generate_mutations

        # Create a mini-dataset from top hits for generation
        hit_dataset = MoleculeDataset(
            records=list(top_hits), name="top_hits",
        )

        generated = generate_mutations(
            hit_dataset, n_molecules=n_generate,
            n_mutations_per_mol=1, seed=seed,
        )

        if generated.valid_records:
            # Score generated molecules
            generated = compute_properties_dataset(generated)
            generated = compute_fingerprints_dataset(generated)
            filter_dataset(generated, lipinski=True, pains=True)
            generated = compute_sa_score_dataset(generated)
            generated = compute_drug_likeness_dataset(generated)
            generated = compute_composite_score_dataset(generated)

            stages["phase3_generation"] = summarize_generation(dataset, generated)

            # Merge generated into main dataset
            dataset.records.extend(generated.records)
    else:
        stages["phase3_generation"] = {"n_generated": 0, "note": "no top hits for generation"}

    # ── Phase 4: 3D Prep for top candidates ──
    logger.info("Phase 4: Preparing 3D for top candidates...")
    all_ranked = rank_molecules(dataset, "composite_score")
    top_candidates = [rec for _, rec, _ in all_ranked[:min(top_n, len(all_ranked))]]

    if top_candidates:
        from drugflow.phase4.ligand_prep.preparation import prepare_ligand

        n_prepared = 0
        for rec in top_candidates:
            if rec.mol is not None:
                try:
                    rec.mol = prepare_ligand(rec.mol, n_confs=5, optimize=True, seed=seed)
                    n_prepared += 1
                except Exception as e:
                    logger.debug(f"3D prep failed for {rec.source_id}: {e}")

        stages["phase4_3d_prep"] = {
            "n_candidates": len(top_candidates),
            "n_prepared_3d": n_prepared,
        }

    # ── Export ──
    logger.info("Exporting results...")
    final_dataset = MoleculeDataset(
        records=list(top_candidates) if top_candidates else dataset.valid_records[:top_n],
        name="hit_to_lead_results",
    )

    export_results_csv(final_dataset, os.path.join(output_dir, "ranked_candidates.csv"))
    write_file(dataset, os.path.join(output_dir, "all_results.csv"))

    report = create_pipeline_report(
        stages,
        metadata={
            "workflow": "hit_to_lead",
            "input": input_path,
            "model": model_path,
            "n_generate": n_generate,
            "top_n": top_n,
        },
    )
    export_project_json(report, os.path.join(output_dir, "report_summary.json"))

    logger.info(f"Hit-to-lead pipeline complete. Results in {output_dir}")
    return report
