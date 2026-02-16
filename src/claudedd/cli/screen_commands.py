"""CLI commands for virtual screening."""

import click


@click.group()
def screen():
    """Virtual screening: substructure, pharmacophore, similarity."""
    pass


@screen.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input molecule file.")
@click.option("--pattern", "-p", required=True,
              help="SMARTS or SMILES pattern for substructure search.")
@click.option("--output", "-o", type=click.Path(), help="Output file for hits.")
@click.option("--exclude", is_flag=True, default=False,
              help="Select molecules that do NOT match.")
@click.option("--count", is_flag=True, default=False,
              help="Also count number of matches per molecule.")
@click.pass_context
def substructure(ctx, input_path, pattern, output, exclude, count):
    """Screen by substructure (SMARTS/SMILES) pattern matching."""
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase2.screening.substructure import screen_substructure
    from claudedd.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    click.echo(f"Screening {len(dataset.valid_records)} molecules for: {pattern}")
    hits = screen_substructure(
        dataset, pattern=pattern, exclude=exclude, count_matches=count,
    )

    click.echo(f"\nHits: {len(hits)} / {len(dataset.valid_records)}")

    # Show a few hits
    for rec in hits.valid_records[:5]:
        smi = rec.canonical_smiles[:50]
        n_matches = rec.properties.get("substruct_match_count", "")
        extra = f" ({n_matches} matches)" if n_matches else ""
        click.echo(f"  {rec.source_id or rec.record_id}: {smi}{extra}")
    if len(hits) > 5:
        click.echo(f"  ... and {len(hits) - 5} more")

    if output:
        write_file(hits, output)
        click.echo(f"Saved hits to {output}")


@screen.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input molecule file.")
@click.option("--features", "-f", required=True,
              help="Feature requirements: 'Donor:1,Acceptor:2,Aromatic:1'")
@click.option("--output", "-o", type=click.Path(), help="Output file for hits.")
@click.option("--match-all/--match-any", default=True,
              help="Require ALL features or ANY feature.")
@click.option("--list-features", is_flag=True, default=False,
              help="List available pharmacophore feature families.")
@click.pass_context
def pharmacophore(ctx, input_path, features, output, match_all, list_features):
    """Screen by pharmacophore feature requirements."""
    from claudedd.phase2.screening.pharmacophore import (
        screen_pharmacophore,
        parse_feature_string,
        get_available_feature_families,
    )

    if list_features:
        families = get_available_feature_families()
        click.echo("Available pharmacophore feature families:")
        for f in families:
            click.echo(f"  {f}")
        return

    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    required = parse_feature_string(features)
    click.echo(f"Requirements: {required}")

    hits = screen_pharmacophore(
        dataset, required_features=required, match_all=match_all,
    )

    mode = "ALL" if match_all else "ANY"
    click.echo(f"\nHits ({mode} match): {len(hits)} / {len(dataset.valid_records)}")

    for rec in hits.valid_records[:5]:
        click.echo(f"  {rec.source_id or rec.record_id}: {rec.canonical_smiles[:50]}")
    if len(hits) > 5:
        click.echo(f"  ... and {len(hits) - 5} more")

    if output:
        write_file(hits, output)
        click.echo(f"Saved hits to {output}")


@screen.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Library molecule file.")
@click.option("--reference", "-r", required=True,
              type=click.Path(exists=True), help="Reference actives file.")
@click.option("--output", "-o", type=click.Path(), help="Output file for hits.")
@click.option("--threshold", "-t", type=float, default=0.7,
              help="Similarity threshold (0-1).")
@click.option("--metric", default="tanimoto",
              type=click.Choice(["tanimoto", "dice", "cosine"]),
              help="Similarity metric.")
@click.option("--aggregation", default="max",
              type=click.Choice(["max", "mean"]),
              help="How to aggregate against multiple references.")
@click.pass_context
def similarity(ctx, input_path, reference, output, threshold, metric, aggregation):
    """Screen by fingerprint similarity to reference actives."""
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from claudedd.phase2.screening.similarity_screen import screen_similarity
    from claudedd.phase1.data.writers import write_file

    click.echo(f"Loading library: {input_path}")
    library = load_file(path=input_path)
    library = validate_dataset(library)
    library = compute_fingerprints_dataset(library)

    click.echo(f"Loading references: {reference}")
    ref_ds = load_file(path=reference)
    ref_ds = validate_dataset(ref_ds)
    ref_ds = compute_fingerprints_dataset(ref_ds)

    click.echo(
        f"Screening {len(library.valid_records)} molecules "
        f"vs {len(ref_ds.valid_records)} references "
        f"(threshold={threshold}, metric={metric})"
    )

    hits = screen_similarity(
        library, ref_ds,
        threshold=threshold, metric=metric, aggregation=aggregation,
    )

    click.echo(f"\nHits: {len(hits)} / {len(library.valid_records)}")

    for rec in hits.valid_records[:5]:
        sim = rec.properties.get("sim_screen_max", 0)
        click.echo(
            f"  {rec.source_id or rec.record_id}: "
            f"sim={sim:.4f} {rec.canonical_smiles[:40]}"
        )
    if len(hits) > 5:
        click.echo(f"  ... and {len(hits) - 5} more")

    if output:
        write_file(hits, output)
        click.echo(f"Saved hits to {output}")
