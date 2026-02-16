"""CLI commands for molecular analysis."""

import click


@click.group()
def analyze():
    """Compute properties, descriptors, fingerprints, and apply filters."""
    pass


@analyze.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", type=click.Path(), help="Output file.")
@click.pass_context
def properties(ctx, input_path, output):
    """Compute physicochemical properties."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    click.echo("Computing properties...")
    dataset = compute_properties_dataset(dataset)

    # Show summary of first molecule
    if dataset.valid_records:
        rec = dataset.valid_records[0]
        click.echo(f"\nSample ({rec.source_id or rec.smiles[:30]}):")
        for key in ["MolWt", "LogP", "TPSA", "HBD", "HBA", "QED"]:
            if key in rec.properties:
                val = rec.properties[key]
                click.echo(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")

    if output:
        write_file(dataset, output)
        click.echo(f"\nSaved to {output}")


@analyze.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", type=click.Path(), help="Output file.")
@click.option("--all", "compute_all", is_flag=True, default=False,
              help="Compute all ~217 RDKit descriptors.")
@click.option("--names", default=None,
              help="Comma-separated list of descriptor names.")
@click.option("--list-available", is_flag=True, default=False,
              help="List all available descriptor names.")
@click.pass_context
def descriptors(ctx, input_path, output, compute_all, names, list_available):
    """Compute molecular descriptors."""
    from drugflow.phase1.analysis.descriptors import (
        compute_descriptors_dataset,
        get_available_descriptors,
    )

    if list_available:
        available = get_available_descriptors()
        click.echo(f"Available descriptors ({len(available)}):")
        for name in available:
            click.echo(f"  {name}")
        return

    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    desc_names = names.split(",") if names else None

    click.echo("Computing descriptors...")
    dataset = compute_descriptors_dataset(
        dataset, descriptor_names=desc_names, compute_all=compute_all,
    )

    if dataset.valid_records:
        rec = dataset.valid_records[0]
        click.echo(f"Computed {len(rec.descriptors)} descriptors per molecule")

    if output:
        write_file(dataset, output)
        click.echo(f"Saved to {output}")


@analyze.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", type=click.Path(), help="Output file (numpy .npz).")
@click.option("--type", "-t", "fp_types", multiple=True,
              default=["morgan"],
              help="Fingerprint type (morgan/maccs/rdkit/atom_pair/topological_torsion).")
@click.option("--radius", type=int, default=2, help="Morgan radius.")
@click.option("--nbits", type=int, default=2048, help="Bit vector length.")
@click.pass_context
def fingerprints(ctx, input_path, output, fp_types, radius, nbits):
    """Compute molecular fingerprints."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from drugflow.phase1.data.writers import write_file
    import numpy as np

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    # Build fp_types config
    fp_config = {}
    for ft in fp_types:
        if ft == "morgan":
            fp_config[f"morgan_r{radius}_{nbits}"] = {
                "type": "morgan", "radius": radius, "nbits": nbits,
            }
        elif ft == "maccs":
            fp_config["maccs"] = {"type": "maccs"}
        else:
            fp_config[ft] = {"type": ft, "nbits": nbits}

    click.echo(f"Computing fingerprints: {list(fp_config.keys())}...")
    dataset = compute_fingerprints_dataset(dataset, fp_types=fp_config)

    if dataset.valid_records:
        rec = dataset.valid_records[0]
        click.echo(f"Fingerprint types: {list(rec.fingerprints.keys())}")

    if output:
        if output.endswith(".npz"):
            # Save as numpy archive
            fps = {}
            for fp_name in fp_config:
                arr = np.array([
                    r.fingerprints[fp_name]
                    for r in dataset.valid_records
                    if fp_name in r.fingerprints
                ])
                fps[fp_name] = arr
            np.savez(output, **fps)
            click.echo(f"Saved fingerprints to {output}")
        else:
            write_file(dataset, output)
            click.echo(f"Saved to {output}")


@analyze.command("filter")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", type=click.Path(),
              help="Output file (passing molecules).")
@click.option("--rejected", type=click.Path(),
              help="Output file for rejected molecules.")
@click.option("--lipinski/--no-lipinski", default=True)
@click.option("--veber/--no-veber", default=False)
@click.option("--pains/--no-pains", default=True)
@click.option("--brenk/--no-brenk", default=False)
@click.option("--max-violations", type=int, default=1,
              help="Max Lipinski violations allowed.")
@click.option("--summary", "show_summary", is_flag=True, default=True,
              help="Print filter statistics.")
@click.pass_context
def filter_cmd(ctx, input_path, output, rejected, lipinski, veber,
               pains, brenk, max_violations, show_summary):
    """Apply drug-likeness filters."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.filters import filter_dataset
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)
    dataset = compute_properties_dataset(dataset)

    click.echo("Applying filters...")
    filtered_ds = filter_dataset(
        dataset,
        lipinski=lipinski, veber=veber, pains=pains, brenk=brenk,
        max_lipinski_violations=max_violations,
        remove_failures=bool(output),
    )

    if show_summary:
        active_filters = []
        if lipinski:
            active_filters.append("Lipinski")
        if veber:
            active_filters.append("Veber")
        if pains:
            active_filters.append("PAINS")
        if brenk:
            active_filters.append("Brenk")

        click.echo(f"\nFilter Summary (active: {', '.join(active_filters)}):")
        click.echo(f"  Total molecules: {len(dataset)}")

        for filt_name in ["lipinski", "veber", "pains", "brenk"]:
            key = f"{filt_name}_pass"
            passing = sum(1 for r in dataset.valid_records
                          if r.properties.get(key) is True)
            failing = sum(1 for r in dataset.valid_records
                          if r.properties.get(key) is False)
            if passing + failing > 0:
                click.echo(f"  {filt_name.upper()}: {passing} pass, {failing} fail")

        if output:
            click.echo(f"  Passing molecules: {len(filtered_ds)}")

    if output:
        write_file(filtered_ds, output)
        click.echo(f"Saved passing molecules to {output}")

    if rejected:
        from drugflow.core.models import MoleculeDataset, MoleculeStatus
        rejected_recs = [r for r in dataset.records
                         if r.status == MoleculeStatus.FILTERED]
        if rejected_recs:
            rejected_ds = MoleculeDataset(records=rejected_recs)
            write_file(rejected_ds, rejected)
            click.echo(f"Saved {len(rejected_recs)} rejected molecules to {rejected}")


@analyze.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--query-smiles", default=None, help="SMILES of query molecule.")
@click.option("--fp-type", default="morgan",
              help="Fingerprint type for comparison.")
@click.option("--metric", default="tanimoto", help="Similarity metric.")
@click.option("--top-k", type=int, default=10, help="Return top K similar.")
@click.option("--output", "-o", type=click.Path(), help="Output file.")
@click.pass_context
def similarity(ctx, input_path, query_smiles, fp_type, metric, top_k, output):
    """Compute molecular similarity."""
    from rdkit import Chem
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.fingerprints import compute_fingerprints_dataset, compute_morgan, compute_maccs
    from drugflow.phase1.analysis.similarity import find_nearest_neighbors, diversity_analysis
    from drugflow.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    # Compute fingerprints
    fp_name = f"{fp_type}_r2_2048" if fp_type == "morgan" else fp_type
    fp_config = {fp_name: {"type": fp_type, "radius": 2, "nbits": 2048}}
    dataset = compute_fingerprints_dataset(dataset, fp_types=fp_config)

    if query_smiles:
        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            raise click.UsageError(f"Invalid query SMILES: {query_smiles}")

        if fp_type == "morgan":
            query_fp = compute_morgan(query_mol)
        else:
            query_fp = compute_maccs(query_mol)

        click.echo(f"\nTop-{top_k} similar to: {query_smiles}")
        neighbors = find_nearest_neighbors(
            query_fp, dataset, fp_type=fp_name,
            top_k=top_k, metric=metric,
        )
        click.echo(f"{'Rank':<6}{'Similarity':<12}{'SMILES'}")
        click.echo("-" * 60)
        for rank, (idx, sim, smi) in enumerate(neighbors, 1):
            click.echo(f"{rank:<6}{sim:<12.4f}{smi[:50]}")
    else:
        click.echo("Computing dataset diversity...")
        stats = diversity_analysis(dataset, fp_type=fp_name)
        click.echo(f"\nDiversity Analysis ({stats['n_molecules']} molecules):")
        click.echo(f"  Mean similarity:   {stats['mean_similarity']:.4f}")
        click.echo(f"  Median similarity: {stats['median_similarity']:.4f}")
        click.echo(f"  Diversity score:   {stats['diversity_score']:.4f}")
        click.echo(f"  Min similarity:    {stats['min_similarity']:.4f}")
        click.echo(f"  Max similarity:    {stats['max_similarity']:.4f}")

    if output:
        write_file(dataset, output)
        click.echo(f"Saved to {output}")


@analyze.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", type=click.Path(), help="Output scored file.")
@click.option("--rank", is_flag=True, default=False,
              help="Rank molecules by composite score.")
@click.option("--top-n", type=int, default=50,
              help="Show top N results when ranking.")
@click.option("--activity-weight", type=float, default=0.4,
              help="Weight for predicted activity.")
@click.option("--dl-weight", type=float, default=0.3,
              help="Weight for drug-likeness.")
@click.option("--sa-weight", type=float, default=0.3,
              help="Weight for synthetic accessibility.")
@click.pass_context
def score(ctx, input_path, output, rank, top_n, activity_weight, dl_weight, sa_weight):
    """Compute composite scores combining activity, drug-likeness, and SA."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase1.analysis.filters import filter_dataset
    from drugflow.phase2.scoring.sa_score import compute_sa_score_dataset
    from drugflow.phase2.scoring.drug_likeness import compute_drug_likeness_dataset
    from drugflow.phase2.scoring.multi_objective import compute_composite_score_dataset
    from drugflow.phase2.scoring.ranking import rank_molecules, export_ranked_results
    from drugflow.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    click.echo("Computing properties...")
    dataset = compute_properties_dataset(dataset)

    click.echo("Applying filters...")
    dataset = filter_dataset(dataset, lipinski=True, pains=True)

    click.echo("Computing SA scores...")
    dataset = compute_sa_score_dataset(dataset)

    click.echo("Computing drug-likeness...")
    dataset = compute_drug_likeness_dataset(dataset)

    click.echo("Computing composite scores...")
    weights = {
        "predicted_activity": activity_weight,
        "drug_likeness": dl_weight,
        "sa_score": sa_weight,
    }
    dataset = compute_composite_score_dataset(dataset, weights=weights)

    if rank:
        ranked = rank_molecules(dataset, "composite_score")
        click.echo(f"\nTop {min(top_n, len(ranked))} molecules:")
        click.echo(f"{'Rank':<6}{'Score':<10}{'SA':<8}{'DL':<8}{'SMILES'}")
        click.echo("-" * 70)
        for r, (idx, rec, scr) in enumerate(ranked[:top_n], 1):
            sa = rec.properties.get("sa_score", 0)
            dl = rec.properties.get("drug_likeness_score", 0)
            smi = rec.canonical_smiles[:40]
            click.echo(f"{r:<6}{scr:<10.4f}{sa:<8.2f}{dl:<8.4f}{smi}")

        if output:
            export_ranked_results(dataset, output, top_n=top_n)
            click.echo(f"\nRanked results saved to {output}")
    elif output:
        write_file(dataset, output)
        click.echo(f"Scored results saved to {output}")
