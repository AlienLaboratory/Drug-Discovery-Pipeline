"""CLI commands for running predefined pipeline workflows."""

import os
from pathlib import Path

import click


@click.group()
def pipeline():
    """Run predefined pipeline workflows."""
    pass


@pipeline.command("quick-profile")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output-dir", "-o", default="./output/profile",
              type=click.Path(), help="Output directory.")
@click.option("--smiles-col", default="smiles",
              help="SMILES column for CSV.")
@click.option("--id-col", default=None, help="ID column for CSV.")
@click.pass_context
def quick_profile(ctx, input_path, output_dir, smiles_col, id_col):
    """Quick molecular profiling: load, analyze, filter, visualize."""
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase1.data.standardizer import standardize_dataset
    from claudedd.phase1.data.writers import write_file
    from claudedd.phase1.analysis.properties import compute_properties_dataset
    from claudedd.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from claudedd.phase1.analysis.filters import filter_dataset
    from claudedd.phase1.analysis.similarity import diversity_analysis
    from claudedd.phase1.visualization.structure import draw_molecule_grid
    from claudedd.phase1.visualization.distributions import (
        plot_property_distributions,
        plot_property_scatter,
    )
    from claudedd.phase1.visualization.chemical_space import plot_chemical_space_pca

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Load & Validate
    click.echo("=" * 60)
    click.echo("ClaudeDD Quick Profile")
    click.echo("=" * 60)

    click.echo(f"\n[1/7] Loading {input_path}...")
    dataset = load_file(
        path=input_path, smiles_column=smiles_col, id_column=id_col,
    )
    click.echo(f"  Loaded {len(dataset)} molecules")

    click.echo("[2/7] Validating & standardizing...")
    dataset = validate_dataset(dataset)
    dataset = standardize_dataset(dataset)
    summary = dataset.summary()
    click.echo(f"  Valid: {summary['valid']}, Failed: {summary['failed']}")

    # Step 2: Properties
    click.echo("[3/7] Computing properties...")
    dataset = compute_properties_dataset(dataset)

    # Step 3: Fingerprints
    click.echo("[4/7] Computing fingerprints...")
    dataset = compute_fingerprints_dataset(dataset)

    # Step 4: Filters
    click.echo("[5/7] Applying drug-likeness filters...")
    filter_dataset(dataset, lipinski=True, pains=True)

    # Print filter summary
    lip_pass = sum(1 for r in dataset.valid_records
                   if r.properties.get("lipinski_pass") is True)
    lip_fail = sum(1 for r in dataset.valid_records
                   if r.properties.get("lipinski_pass") is False)
    pains_pass = sum(1 for r in dataset.valid_records
                     if r.properties.get("pains_pass") is True)
    pains_fail = sum(1 for r in dataset.valid_records
                     if r.properties.get("pains_pass") is False)

    click.echo(f"  Lipinski: {lip_pass} pass, {lip_fail} fail")
    click.echo(f"  PAINS: {pains_pass} pass, {pains_fail} fail")

    # Step 5: Diversity
    stats = diversity_analysis(dataset, fp_type="morgan_r2_2048")
    click.echo(f"  Diversity score: {stats['diversity_score']:.4f}")

    # Step 6: Visualization
    click.echo("[6/7] Generating visualizations...")

    draw_molecule_grid(
        dataset,
        os.path.join(output_dir, "structure_grid.png"),
        mols_per_page=20, cols=5, property_label="QED",
    )

    plot_property_distributions(
        dataset,
        ["MolWt", "LogP", "TPSA", "HBD", "HBA", "QED",
         "NumRotatableBonds", "RingCount", "FractionCSP3"],
        os.path.join(output_dir, "property_distributions.png"),
    )

    plot_property_scatter(
        dataset, "MolWt", "LogP",
        os.path.join(output_dir, "mw_vs_logp.png"),
        color_property="QED",
    )

    plot_chemical_space_pca(
        dataset,
        fp_type="morgan_r2_2048",
        output_path=os.path.join(output_dir, "chemical_space_pca.png"),
        color_property="QED",
    )

    # Step 7: Export
    click.echo("[7/7] Exporting results...")
    write_file(dataset, os.path.join(output_dir, "results.csv"))

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Profile complete! Results in: {output_dir}")
    click.echo(f"  - structure_grid.png")
    click.echo(f"  - property_distributions.png")
    click.echo(f"  - mw_vs_logp.png")
    click.echo(f"  - chemical_space_pca.png")
    click.echo(f"  - results.csv")
    click.echo(f"{'=' * 60}")


@pipeline.command()
@click.option("--config", "-c", required=True,
              type=click.Path(exists=True), help="YAML config file.")
@click.pass_context
def run(ctx, config):
    """Run a complete pipeline from config file."""
    from claudedd.core.config import PipelineConfig
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase1.data.standardizer import standardize_dataset
    from claudedd.phase1.data.writers import write_file
    from claudedd.phase1.analysis.properties import compute_properties_dataset
    from claudedd.phase1.analysis.descriptors import compute_descriptors_dataset
    from claudedd.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from claudedd.phase1.analysis.filters import filter_dataset

    cfg = PipelineConfig.from_yaml(config)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    click.echo(f"Running pipeline: {cfg.name}")

    # Load
    dataset = load_file(
        path=cfg.input_file,
        format=cfg.input_format,
        smiles_column=cfg.smiles_column,
        id_column=cfg.id_column,
    )
    click.echo(f"Loaded {len(dataset)} molecules")

    # Validate
    dataset = validate_dataset(dataset)

    # Standardize
    if cfg.standardize:
        dataset = standardize_dataset(
            dataset,
            strip_salts=cfg.strip_salts,
            neutralize=cfg.neutralize,
            remove_stereo=cfg.remove_stereo,
        )

    # Properties
    dataset = compute_properties_dataset(dataset)

    # Descriptors
    dataset = compute_descriptors_dataset(
        dataset,
        descriptor_names=cfg.descriptors.descriptor_names,
        compute_all=cfg.descriptors.compute_all,
    )

    # Fingerprints
    dataset = compute_fingerprints_dataset(dataset)

    # Filters
    dataset = filter_dataset(
        dataset,
        lipinski=cfg.filters.lipinski,
        veber=cfg.filters.veber,
        pains=cfg.filters.pains,
        brenk=cfg.filters.brenk,
        max_lipinski_violations=cfg.filters.lipinski_max_violations,
    )

    # Export
    output_path = os.path.join(cfg.output_dir, f"{cfg.name}_results.csv")
    write_file(dataset, output_path)
    click.echo(f"Results saved to {output_path}")


@pipeline.command("init-config")
@click.option("--output", "-o", default="./configs/pipeline.yaml",
              type=click.Path(), help="Output config path.")
@click.pass_context
def init_config(ctx, output):
    """Generate a template YAML config file."""
    from claudedd.core.config import PipelineConfig

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig()
    config.to_yaml(output)
    click.echo(f"Template config written to {output}")


@pipeline.command("screen-and-score")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Library molecule file.")
@click.option("--reference", "-r",
              type=click.Path(exists=True), default=None,
              help="Reference actives for similarity screening.")
@click.option("--model", "-m", "model_path",
              type=click.Path(exists=True), default=None,
              help="Trained QSAR model for activity prediction.")
@click.option("--output-dir", "-o", default="./output/screen_score",
              type=click.Path(), help="Output directory.")
@click.option("--similarity-threshold", type=float, default=0.7,
              help="Similarity threshold for screening.")
@click.option("--top-n", type=int, default=50,
              help="Number of top candidates to export.")
@click.pass_context
def screen_and_score(ctx, input_path, reference, model_path, output_dir,
                     similarity_threshold, top_n):
    """Full screen-and-score pipeline: screen, predict, score, rank."""
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase1.data.standardizer import standardize_dataset
    from claudedd.phase1.analysis.properties import compute_properties_dataset
    from claudedd.phase1.analysis.descriptors import compute_descriptors_dataset
    from claudedd.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from claudedd.phase1.analysis.filters import filter_dataset
    from claudedd.phase1.data.writers import write_file

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    click.echo("=" * 60)
    click.echo("ClaudeDD Screen & Score Pipeline")
    click.echo("=" * 60)

    # Step 1: Load & Prepare
    click.echo(f"\n[1/8] Loading {input_path}...")
    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)
    dataset = standardize_dataset(dataset)
    click.echo(f"  {len(dataset.valid_records)} valid molecules")

    # Step 2: Compute properties & fingerprints
    click.echo("[2/8] Computing properties & fingerprints...")
    dataset = compute_properties_dataset(dataset)
    dataset = compute_fingerprints_dataset(dataset)

    # Step 3: Filters
    click.echo("[3/8] Applying drug-likeness filters...")
    dataset = filter_dataset(dataset, lipinski=True, pains=True)

    # Step 4: Similarity screening (optional)
    if reference:
        click.echo(f"[4/8] Similarity screening vs {reference}...")
        from claudedd.phase2.screening.similarity_screen import screen_similarity

        ref_ds = load_file(path=reference)
        ref_ds = validate_dataset(ref_ds)
        ref_ds = compute_fingerprints_dataset(ref_ds)

        dataset = screen_similarity(
            dataset, ref_ds, threshold=similarity_threshold,
        )
        click.echo(f"  {len(dataset.valid_records)} hits above threshold")
    else:
        click.echo("[4/8] Skipping similarity screening (no reference provided)")

    # Step 5: QSAR prediction (optional)
    if model_path:
        click.echo(f"[5/8] Predicting activity with {model_path}...")
        from claudedd.phase2.qsar.persistence import load_model
        from claudedd.phase2.qsar.prediction import predict_dataset

        qsar_model = load_model(model_path)

        # Compute descriptors if model needs them
        if not qsar_model.feature_names[0].startswith(("morgan_", "maccs_")):
            dataset = compute_descriptors_dataset(dataset)
            predict_dataset(qsar_model, dataset, feature_source="descriptors")
        else:
            fp_type = qsar_model.feature_names[0].rsplit("_bit_", 1)[0]
            predict_dataset(qsar_model, dataset, feature_source="fingerprints", fp_type=fp_type)
    else:
        click.echo("[5/8] Skipping QSAR prediction (no model provided)")

    # Step 6: SA Score
    click.echo("[6/8] Computing SA scores...")
    from claudedd.phase2.scoring.sa_score import compute_sa_score_dataset
    dataset = compute_sa_score_dataset(dataset)

    # Step 7: Drug-likeness & Composite scoring
    click.echo("[7/8] Computing drug-likeness & composite scores...")
    from claudedd.phase2.scoring.drug_likeness import compute_drug_likeness_dataset
    from claudedd.phase2.scoring.multi_objective import compute_composite_score_dataset
    from claudedd.phase2.scoring.ranking import (
        rank_molecules, export_ranked_results, flag_candidates,
    )

    dataset = compute_drug_likeness_dataset(dataset)
    dataset = compute_composite_score_dataset(dataset)

    # Step 8: Rank & Export
    click.echo("[8/8] Ranking and exporting results...")
    dataset = flag_candidates(dataset)

    # Export all results
    write_file(dataset, os.path.join(output_dir, "all_results.csv"))

    # Export ranked top-N
    export_ranked_results(
        dataset,
        os.path.join(output_dir, "ranked_candidates.csv"),
        top_n=top_n,
    )

    # Summary
    ranked = rank_molecules(dataset, "composite_score")
    n_candidates = sum(
        1 for r in dataset.valid_records
        if r.properties.get("candidate_flag")
    )

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Pipeline complete!")
    click.echo(f"  Total molecules scored: {len(dataset.valid_records)}")
    click.echo(f"  Candidates flagged: {n_candidates}")
    click.echo(f"  Output directory: {output_dir}")
    click.echo(f"  - all_results.csv")
    click.echo(f"  - ranked_candidates.csv")
    if ranked:
        top = ranked[0]
        click.echo(
            f"  Top molecule: {top[1].source_id or top[1].canonical_smiles[:30]} "
            f"(score={top[2]:.4f})"
        )
    click.echo(f"{'=' * 60}")
