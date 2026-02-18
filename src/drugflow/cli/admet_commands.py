"""CLI commands for ADMET prediction."""

import click


@click.group()
def admet():
    """ADMET property predictions: absorption, distribution, metabolism, excretion, toxicity."""
    pass


@admet.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True),
              help="Input molecules file (CSV/SDF/SMI).")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output CSV file with ADMET predictions.")
@click.option("--weights", type=str, default=None,
              help='Custom scoring weights as JSON, e.g. \'{"toxicity": 0.4}\'.')
@click.pass_context
def predict(ctx, input_path, output_path, weights):
    """Predict ADMET properties for a set of molecules.

    Computes rule-based ADMET predictions (absorption, distribution,
    metabolism, excretion, toxicity) plus aggregate ADMET score and
    risk classification for each molecule.
    """
    import json
    import os

    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase5.admet.pipeline import predict_admet_dataset
    from drugflow.phase1.data.writers import write_file

    click.echo("=" * 60)
    click.echo("DrugFlow ADMET Prediction")
    click.echo("=" * 60)

    # Load and prepare
    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)
    dataset = compute_properties_dataset(dataset)

    click.echo(f"Loaded {len(dataset.valid_records)} valid molecules from {input_path}")

    # Parse custom weights
    weight_dict = None
    if weights:
        try:
            weight_dict = json.loads(weights)
        except json.JSONDecodeError:
            click.echo("Warning: could not parse weights JSON, using defaults.", err=True)

    # Run predictions
    dataset = predict_admet_dataset(dataset, weights=weight_dict)

    # Print summary
    scored = [r for r in dataset.valid_records if "admet_score" in r.properties]
    if scored:
        scores = [r.properties["admet_score"] for r in scored]
        click.echo(f"\nADMET Score: mean={sum(scores)/len(scores):.3f}, "
                    f"min={min(scores):.3f}, max={max(scores):.3f}")

        # Classification counts
        classes = {}
        for r in scored:
            cls = r.properties.get("admet_class", "unknown")
            classes[cls] = classes.get(cls, 0) + 1

        click.echo("\nClassification:")
        for cls in ["favorable", "moderate", "poor"]:
            n = classes.get(cls, 0)
            pct = 100 * n / len(scored) if scored else 0
            click.echo(f"  {cls}: {n} ({pct:.1f}%)")

        # Risk flag summary
        total_red = sum(r.properties.get("admet_n_red_flags", 0) for r in scored)
        total_yellow = sum(r.properties.get("admet_n_yellow_flags", 0) for r in scored)
        total_green = sum(r.properties.get("admet_n_green_flags", 0) for r in scored)
        click.echo(f"\nTotal risk flags: {total_green} green, "
                    f"{total_yellow} yellow, {total_red} red")

        # Top 5 by ADMET score
        sorted_recs = sorted(scored, key=lambda r: r.properties["admet_score"], reverse=True)
        click.echo("\nTop 5 by ADMET score:")
        for rec in sorted_recs[:5]:
            sid = rec.source_id or rec.record_id
            sc = rec.properties["admet_score"]
            cls = rec.properties.get("admet_class", "?")
            red = rec.properties.get("admet_n_red_flags", 0)
            click.echo(f"  {sid}: score={sc:.3f}, class={cls}, red_flags={red}")

    # Save output
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        write_file(dataset, output_path)
        click.echo(f"\nSaved ADMET results to {output_path}")

    click.echo("=" * 60)


@admet.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True),
              help="Input molecules file (with ADMET predictions).")
@click.pass_context
def summary(ctx, input_path):
    """Show ADMET summary statistics for a molecule set.

    Displays distribution of risk flags, ADMET scores, and
    identifies the most common liabilities.
    """
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase5.admet.pipeline import predict_admet_dataset

    click.echo("=" * 60)
    click.echo("DrugFlow ADMET Summary")
    click.echo("=" * 60)

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)
    dataset = compute_properties_dataset(dataset)
    dataset = predict_admet_dataset(dataset)

    scored = [r for r in dataset.valid_records if "admet_score" in r.properties]
    if not scored:
        click.echo("No molecules with ADMET scores found.")
        return

    click.echo(f"\nTotal molecules: {len(scored)}")

    # Domain sub-score averages
    domains = ["absorption", "distribution", "metabolism", "excretion", "toxicity"]
    click.echo("\nDomain scores (mean):")
    for domain in domains:
        key = f"admet_{domain}_score"
        vals = [r.properties.get(key, 0) for r in scored if key in r.properties]
        if vals:
            click.echo(f"  {domain:15s}: {sum(vals)/len(vals):.3f}")

    # Most common liabilities
    click.echo("\nMost common liabilities:")
    liability_counts = {}

    for rec in scored:
        # Check hERG
        if rec.properties.get("admet_herg_risk") == "red":
            liability_counts["hERG risk"] = liability_counts.get("hERG risk", 0) + 1
        # Check AMES
        if rec.properties.get("admet_ames_risk") in ("red", "yellow"):
            liability_counts["AMES alerts"] = liability_counts.get("AMES alerts", 0) + 1
        # Check hepatotox
        if rec.properties.get("admet_hepatotox_risk") in ("red", "yellow"):
            liability_counts["Hepatotox alerts"] = liability_counts.get("Hepatotox alerts", 0) + 1
        # Check poor HIA
        if rec.properties.get("admet_hia_class") == "low":
            liability_counts["Low HIA"] = liability_counts.get("Low HIA", 0) + 1
        # Check Pgp
        if rec.properties.get("admet_pgp_substrate") is True:
            liability_counts["Pgp substrate"] = liability_counts.get("Pgp substrate", 0) + 1
        # Check CYP
        if rec.properties.get("admet_cyp_risk") in ("red", "yellow"):
            liability_counts["CYP inhibition"] = liability_counts.get("CYP inhibition", 0) + 1

    for liability, count in sorted(liability_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(scored)
        click.echo(f"  {liability:20s}: {count:4d} ({pct:.1f}%)")

    click.echo("=" * 60)
