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
