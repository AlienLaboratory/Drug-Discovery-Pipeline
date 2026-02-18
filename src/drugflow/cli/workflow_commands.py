"""CLI commands for end-to-end drug discovery workflows."""

import click


@click.group()
def workflow():
    """End-to-end drug discovery workflows connecting all phases."""
    pass


@workflow.command("hit-to-lead")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input molecules file.")
@click.option("--output-dir", "-o", "output_dir", default="./output/hit2lead",
              type=click.Path(), help="Output directory.")
@click.option("--model", "-m", "model_path",
              type=click.Path(exists=True), default=None,
              help="Optional trained QSAR model for scoring.")
@click.option("--n-generate", default=50, type=int,
              help="Number of analogs to generate from top hits.")
@click.option("--top-n", default=20, type=int,
              help="Number of top candidates to export.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.pass_context
def hit_to_lead(ctx, input_path, output_dir, model_path, n_generate, top_n, seed):
    """Hit-to-Lead: profile -> score -> generate analogs -> rank.

    Full pipeline: load molecules, compute properties, apply filters,
    score with SA/drug-likeness, generate analogs from top hits via
    mutations, rescore, prepare 3D for top candidates, and export.
    """
    from drugflow.phase5.workflows.hit_to_lead import run_hit_to_lead

    click.echo("=" * 60)
    click.echo("DrugFlow Hit-to-Lead Workflow")
    click.echo("=" * 60)

    report = run_hit_to_lead(
        input_path, output_dir,
        model_path=model_path,
        n_generate=n_generate,
        top_n=top_n, seed=seed,
    )

    # Print summary
    stages = report.get("stages", {})
    if "phase1_profile" in stages:
        p1 = stages["phase1_profile"]
        click.echo(f"\n[Phase 1] Input: {p1.get('total_records', '?')} molecules, "
                    f"{p1.get('valid_records', '?')} valid")

    if "phase2_scoring" in stages:
        p2 = stages["phase2_scoring"]
        click.echo(f"[Phase 2] Scored: {p2.get('n_scored', 0)} molecules")

    if "phase3_generation" in stages:
        p3 = stages["phase3_generation"]
        click.echo(f"[Phase 3] Generated: {p3.get('n_generated', p3.get('n_valid', 0))} analogs")

    if "phase4_3d_prep" in stages:
        p4 = stages["phase4_3d_prep"]
        click.echo(f"[Phase 4] 3D prepared: {p4.get('n_prepared_3d', 0)} candidates")

    click.echo(f"\nResults saved to: {output_dir}")
    click.echo(f"  - ranked_candidates.csv")
    click.echo(f"  - all_results.csv")
    click.echo(f"  - report_summary.json")
    click.echo("=" * 60)


@workflow.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Seed molecules file.")
@click.option("--output-dir", "-o", "output_dir", default="./output/denovo",
              type=click.Path(), help="Output directory.")
@click.option("--strategy", default="brics",
              type=click.Choice(["brics", "mutate"]),
              help="Generation strategy.")
@click.option("--model", "-m", "model_path",
              type=click.Path(exists=True), default=None,
              help="Optional QSAR model for scoring.")
@click.option("-n", "n_generate", default=100, type=int,
              help="Number of molecules to generate.")
@click.option("--top-n", default=30, type=int,
              help="Number of top candidates to keep.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.pass_context
def denovo(ctx, input_path, output_dir, strategy, model_path, n_generate, top_n, seed):
    """De Novo Design: generate novel molecules from seeds.

    Generate molecules using BRICS fragment recombination or random
    mutations, validate and score them, prepare 3D structures for
    top candidates, and export ranked results.
    """
    from drugflow.phase5.workflows.denovo_design import run_denovo_design

    click.echo("=" * 60)
    click.echo("DrugFlow De Novo Design Workflow")
    click.echo("=" * 60)

    report = run_denovo_design(
        input_path, output_dir,
        strategy=strategy,
        model_path=model_path,
        n_generate=n_generate,
        top_n=top_n, seed=seed,
    )

    stages = report.get("stages", {})
    if "phase1_seeds" in stages:
        p1 = stages["phase1_seeds"]
        click.echo(f"\n[Phase 1] Seeds: {p1.get('valid_records', '?')} molecules")

    if "phase3_generation" in stages:
        p3 = stages["phase3_generation"]
        click.echo(f"[Phase 3] Generated: {p3.get('n_valid', 0)} valid molecules "
                    f"({p3.get('novelty_rate', 0):.0%} novel)")

    if "phase2_scoring" in stages:
        p2 = stages["phase2_scoring"]
        click.echo(f"[Phase 2] Scored: {p2.get('n_scored', 0)} molecules")

    if "phase4_3d_prep" in stages:
        p4 = stages["phase4_3d_prep"]
        click.echo(f"[Phase 4] 3D prepared: {p4.get('n_prepared_3d', 0)} candidates")

    click.echo(f"\nResults saved to: {output_dir}")
    click.echo(f"  - generated_scored.csv")
    click.echo(f"  - all_generated.csv")
    click.echo(f"  - report_summary.json")
    click.echo("=" * 60)


@workflow.command()
@click.option("--lead", required=True,
              help="SMILES string of the lead molecule.")
@click.option("--output-dir", "-o", "output_dir", default="./output/optimize",
              type=click.Path(), help="Output directory.")
@click.option("--n-analogs", default=50, type=int,
              help="Number of analogs to generate.")
@click.option("--top-n", default=20, type=int,
              help="Number of top candidates.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.pass_context
def optimize(ctx, lead, output_dir, n_analogs, top_n, seed):
    """Lead Optimization: generate and rank analogs of a lead compound.

    Takes a single lead molecule SMILES, generates analogs via
    mutations and scaffold decoration, scores them, prepares 3D
    structures, shape-screens against the lead, and exports
    optimized candidates.
    """
    from drugflow.phase5.workflows.lead_optimization import run_lead_optimization

    click.echo("=" * 60)
    click.echo("DrugFlow Lead Optimization Workflow")
    click.echo("=" * 60)
    click.echo(f"Lead: {lead}")

    try:
        report = run_lead_optimization(
            lead, output_dir,
            n_analogs=n_analogs,
            top_n=top_n, seed=seed,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    stages = report.get("stages", {})
    if "phase3_generation" in stages:
        p3 = stages["phase3_generation"]
        click.echo(f"\n[Phase 3] Generated: {p3.get('n_total', 0)} analogs "
                    f"({p3.get('n_mutations', 0)} mutations, "
                    f"{p3.get('n_decorated', 0)} decorated)")

    if "phase2_scoring" in stages:
        p2 = stages["phase2_scoring"]
        click.echo(f"[Phase 2] Scored: {p2.get('n_scored', 0)} molecules")

    if "phase4_3d_prep" in stages:
        p4 = stages["phase4_3d_prep"]
        click.echo(f"[Phase 4] 3D prepared: {p4.get('n_prepared_3d', 0)}, "
                    f"shape-screened vs lead")

    click.echo(f"\nResults saved to: {output_dir}")
    click.echo(f"  - optimized_candidates.csv")
    click.echo(f"  - all_analogs.csv")
    click.echo(f"  - report_summary.json")
    click.echo("=" * 60)


@workflow.command("research-report")
@click.option("--candidates", "-c", required=True,
              type=click.Path(exists=True),
              help="CSV file with novel candidate molecules.")
@click.option("--training-data", "-t", required=True,
              type=click.Path(exists=True),
              help="CSV file with curated training data.")
@click.option("--output-dir", "-o", default="./research_report",
              type=click.Path(),
              help="Output directory for the research report.")
@click.option("--model", "-m", "model_path", default=None,
              type=click.Path(exists=True),
              help="Trained QSAR model (.joblib) for metadata.")
@click.option("--model-comparison", default=None,
              type=click.Path(exists=True),
              help="Model comparison CSV from `model compare`.")
@click.option("--target-name", default="Unknown Target",
              help="Name of the biological target.")
@click.option("--campaign-name", default="Research Campaign",
              help="Name of the research campaign.")
@click.pass_context
def research_report(ctx, candidates, training_data, output_dir,
                    model_path, model_comparison, target_name, campaign_name):
    """Generate a comprehensive research report for a drug discovery campaign.

    Assembles all results into a structured folder with summary,
    candidate CSVs, statistics, and visualizations.
    """
    from drugflow.phase5.reporting.research_report import generate_research_report

    click.echo("=" * 60)
    click.echo("DrugFlow Research Report Generator")
    click.echo("=" * 60)
    click.echo(f"Target: {target_name}")
    click.echo(f"Campaign: {campaign_name}")

    result_dir = generate_research_report(
        candidates_path=candidates,
        training_data_path=training_data,
        output_dir=output_dir,
        model_path=model_path,
        model_comparison_path=model_comparison,
        target_name=target_name,
        campaign_name=campaign_name,
    )

    # List output files
    click.echo(f"\nReport generated at: {result_dir}")
    import os
    for root, dirs, files in os.walk(result_dir):
        level = root.replace(result_dir, "").count(os.sep)
        indent = "  " * level
        folder = os.path.basename(root)
        if level > 0:
            click.echo(f"{indent}{folder}/")
        for f in files:
            click.echo(f"{indent}  - {f}")
    click.echo("=" * 60)


@workflow.command("docking-report")
@click.option("--input-dir", "-i", required=True,
              type=click.Path(exists=True),
              help="Directory containing docking_results.csv (and optionally "
                   "admet_all_candidates.csv, study_report.json).")
@click.option("--output-dir", "-o", default=None,
              type=click.Path(),
              help="Output directory for report. Defaults to INPUT_DIR/report.")
@click.option("--target-name", default="Unknown Target",
              help="Name of the biological target (e.g. 'BCL-2 (PDB: 6O0K)').")
@click.option("--campaign-name", default="Docking Study",
              help="Name of the research campaign.")
@click.pass_context
def docking_report(ctx, input_dir, output_dir, target_name, campaign_name):
    """Generate publication-quality docking results report.

    Produces 8 PNG visualizations, ranked CSV with composite scores,
    enhanced JSON report, text summary, and statistics CSV from a
    completed docking study.
    """
    import os

    from drugflow.phase5.reporting.docking_report import generate_docking_report

    click.echo("=" * 60)
    click.echo("DrugFlow Docking Report Generator")
    click.echo("=" * 60)
    click.echo(f"Target: {target_name}")
    click.echo(f"Campaign: {campaign_name}")
    click.echo(f"Input: {input_dir}")

    # Auto-detect files
    docking_csv = os.path.join(input_dir, "docking_results.csv")
    if not os.path.isfile(docking_csv):
        click.echo(f"ERROR: {docking_csv} not found.", err=True)
        return

    admet_csv = os.path.join(input_dir, "admet_all_candidates.csv")
    if not os.path.isfile(admet_csv):
        admet_csv = None

    study_json = os.path.join(input_dir, "study_report.json")
    if not os.path.isfile(study_json):
        study_json = None

    if output_dir is None:
        output_dir = os.path.join(input_dir, "report")

    click.echo(f"Output: {output_dir}")
    click.echo("")

    result_dir = generate_docking_report(
        docking_results_path=docking_csv,
        output_dir=output_dir,
        admet_all_path=admet_csv,
        study_report_path=study_json,
        target_name=target_name,
        campaign_name=campaign_name,
    )

    # List output files
    click.echo(f"\nReport generated at: {result_dir}")
    for root, dirs, files in os.walk(result_dir):
        level = root.replace(result_dir, "").count(os.sep)
        indent = "  " * level
        folder = os.path.basename(root)
        if level > 0:
            click.echo(f"{indent}{folder}/")
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size_kb = os.path.getsize(fpath) / 1024
            click.echo(f"{indent}  - {f} ({size_kb:.1f} KB)")
    click.echo("=" * 60)
