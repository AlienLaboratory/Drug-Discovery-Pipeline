"""CLI commands for data loading, validation, standardization, and export."""

import click


@click.group()
def data():
    """Data loading, validation, standardization, and export."""
    pass


@data.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input molecular file.")
@click.option("--format", "-f", "fmt", default="auto",
              help="File format (auto/sdf/smi/csv/pdb).")
@click.option("--smiles-col", default="smiles",
              help="SMILES column name for CSV files.")
@click.option("--id-col", default=None, help="ID column name.")
@click.option("--limit", type=int, default=None,
              help="Max molecules to load.")
@click.option("--output", "-o", type=click.Path(), help="Output file path.")
@click.option("--output-format", default="csv", help="Output file format.")
@click.pass_context
def load(ctx, input_path, fmt, smiles_col, id_col, limit, output, output_format):
    """Load molecules from a file, validate, and optionally export."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.data.writers import write_file

    click.echo(f"Loading molecules from {input_path}...")
    dataset = load_file(
        path=input_path, format=fmt,
        smiles_column=smiles_col, id_column=id_col, limit=limit,
    )
    click.echo(f"  Loaded {len(dataset)} molecules.")

    click.echo("Validating...")
    dataset = validate_dataset(dataset)
    summary = dataset.summary()
    click.echo(f"  Valid: {summary['valid']}, Failed: {summary['failed']}")

    if output:
        write_file(dataset, output, format=output_format)
        click.echo(f"  Saved to {output}")


@data.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output file.")
@click.option("--strip-salts/--no-strip-salts", default=True)
@click.option("--neutralize/--no-neutralize", default=True)
@click.option("--canonicalize-tautomers", is_flag=True, default=False)
@click.option("--remove-stereo", is_flag=True, default=False)
@click.pass_context
def standardize(ctx, input_path, output, strip_salts, neutralize,
                canonicalize_tautomers, remove_stereo):
    """Standardize molecules (salts, charges, tautomers)."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.data.standardizer import standardize_dataset
    from drugflow.phase1.data.writers import write_file

    click.echo(f"Loading and validating {input_path}...")
    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    click.echo("Standardizing...")
    dataset = standardize_dataset(
        dataset,
        strip_salts=strip_salts,
        neutralize=neutralize,
        canonicalize_tautomers=canonicalize_tautomers,
        remove_stereo=remove_stereo,
    )

    write_file(dataset, output)
    click.echo(f"Saved standardized molecules to {output}")


@data.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output file.")
@click.option("--format", "-f", "fmt", default="auto",
              help="Output format (csv/sdf/smi/json).")
@click.pass_context
def export(ctx, input_path, output, fmt):
    """Export dataset to a different format."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    write_file(dataset, output, format=fmt)
    click.echo(f"Exported {len(dataset)} molecules to {output}")


@data.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
def info(input_path):
    """Show information about a dataset file."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)
    summary = dataset.summary()

    click.echo(f"File: {input_path}")
    click.echo(f"  Total molecules: {summary['total']}")
    click.echo(f"  Valid: {summary['valid']}")
    click.echo(f"  Failed: {summary['failed']}")

    if dataset.valid_records:
        rec = dataset.valid_records[0]
        if rec.metadata:
            click.echo(f"  Metadata fields: {list(rec.metadata.keys())}")
        if rec.properties:
            click.echo(f"  Property fields: {list(rec.properties.keys())}")


@data.command()
@click.option("--source", required=True,
              type=click.Choice(["chembl", "pubchem"]),
              help="Database source.")
@click.option("--target", default=None, help="ChEMBL target ID.")
@click.option("--query", default=None, help="Search query (PubChem).")
@click.option("--max-results", type=int, default=1000)
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output file.")
@click.pass_context
def fetch(ctx, source, target, query, max_results, output):
    """Fetch molecules from a public database."""
    from drugflow.phase1.data.databases import fetch_chembl_by_target, fetch_pubchem_by_name
    from drugflow.phase1.data.writers import write_file

    if source == "chembl":
        if not target:
            raise click.UsageError("--target is required for ChEMBL")
        click.echo(f"Fetching from ChEMBL (target={target})...")
        dataset = fetch_chembl_by_target(target, max_results=max_results)
    else:
        if not query:
            raise click.UsageError("--query is required for PubChem")
        click.echo(f"Fetching from PubChem (query={query})...")
        dataset = fetch_pubchem_by_name(query, max_results=max_results)

    click.echo(f"  Fetched {len(dataset)} molecules")
    write_file(dataset, output)
    click.echo(f"  Saved to {output}")
