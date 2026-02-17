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


@data.command("fetch-bioactivity")
@click.option("--target", "-t", required=True,
              help="ChEMBL target ID (e.g., CHEMBL4860 for BCL-2).")
@click.option("--activity-types", default="IC50",
              help="Comma-separated activity types: IC50,Ki,Kd,EC50.")
@click.option("--max-results", type=int, default=10000,
              help="Maximum records to fetch.")
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output CSV file path.")
@click.option("--save-raw", type=click.Path(), default=None,
              help="Save raw (pre-curation) data to this CSV.")
@click.pass_context
def fetch_bioactivity(ctx, target, activity_types, max_results, output, save_raw):
    """Fetch bioactivity data from ChEMBL with pagination.

    Downloads activity records for a target protein, stores rich
    metadata (pChEMBL, assay info), and saves as CSV.

    Example: drugflow data fetch-bioactivity -t CHEMBL4860 -o bcl2_raw.csv
    """
    from drugflow.phase1.data.bioactivity import fetch_chembl_bioactivity
    from drugflow.phase1.data.writers import write_file

    types_list = [t.strip() for t in activity_types.split(",")]
    click.echo(f"Fetching from ChEMBL: target={target}, types={types_list}")

    dataset = fetch_chembl_bioactivity(
        target_id=target,
        activity_types=types_list,
        max_results=max_results,
        save_raw=save_raw,
    )

    click.echo(f"  Fetched {len(dataset)} activity records")
    if save_raw:
        click.echo(f"  Raw backup saved to {save_raw}")

    write_file(dataset, output)
    click.echo(f"  Saved to {output}")


@data.command("curate")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input CSV with activity data.")
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output curated CSV.")
@click.option("--relations", default="=",
              help="Allowed relations, comma-separated (default: '=').")
@click.option("--include-less-than", is_flag=True, default=False,
              help="Also include '<' and '<=' relations.")
@click.option("--deduplicate/--no-deduplicate", default=True,
              help="Merge duplicate molecules (default: yes).")
@click.option("--dedup-key", default="canonical_smiles",
              type=click.Choice(["canonical_smiles", "inchikey"]),
              help="Key for deduplication.")
@click.option("--remove-outliers", is_flag=True, default=False,
              help="Remove activity value outliers (IQR method).")
@click.option("--iqr-multiplier", type=float, default=1.5,
              help="IQR multiplier for outlier bounds.")
@click.pass_context
def curate(ctx, input_path, output, relations, include_less_than,
           deduplicate, dedup_key, remove_outliers, iqr_multiplier):
    """Curate bioactivity data: filter, normalize, deduplicate.

    Applies a curation pipeline to raw bioactivity data:
    1. Filter by measurement relation
    2. Normalize units to nM
    3. Deduplicate by canonical SMILES (median)
    4. Optionally remove outliers

    Example: drugflow data curate -i raw.csv -o curated.csv --remove-outliers
    """
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.bioactivity import curate_bioactivity
    from drugflow.phase1.data.writers import write_file

    click.echo(f"[1/3] Loading {input_path}...")
    dataset = load_file(path=input_path)
    click.echo(f"  Loaded {len(dataset)} records")

    if include_less_than:
        allowed = {"=", "<", "<="}
    else:
        allowed = set(r.strip() for r in relations.split(","))

    click.echo(f"[2/3] Curating (relations={allowed}, dedup={deduplicate}, "
               f"outliers={remove_outliers})...")
    curated, stats = curate_bioactivity(
        dataset,
        allowed_relations=allowed,
        deduplicate=deduplicate,
        dedup_key=dedup_key,
        remove_outliers=remove_outliers,
        iqr_multiplier=iqr_multiplier,
    )

    click.echo(f"  Curation summary:")
    click.echo(f"    Input:            {stats.input_count}")
    click.echo(f"    After relations:  {stats.after_relation_filter}")
    click.echo(f"    After units:      {stats.after_unit_filter}")
    click.echo(f"    After dedup:      {stats.after_dedup} "
               f"({stats.duplicates_merged} merged)")
    if remove_outliers:
        click.echo(f"    After outliers:   {stats.after_outlier_removal} "
                   f"({stats.outliers_removed} removed)")
    click.echo(f"    Final output:     {stats.output_count}")

    click.echo(f"[3/3] Saving to {output}...")
    write_file(curated, output)
    click.echo(f"  Done. {stats.output_count} curated records saved.")


@data.command("label")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input curated CSV.")
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output labeled CSV.")
@click.option("--mode", default="binary",
              type=click.Choice(["binary", "ternary"]),
              help="Labeling mode.")
@click.option("--active-threshold", type=float, default=1000.0,
              help="IC50 threshold (nM) for active class. Default 1000.")
@click.option("--inactive-threshold", type=float, default=10000.0,
              help="IC50 threshold (nM) for inactive (ternary only).")
@click.option("--value-key", default="standard_value",
              help="Metadata key for activity value.")
@click.pass_context
def label(ctx, input_path, output, mode, active_threshold,
          inactive_threshold, value_key):
    """Label activity data for QSAR classification.

    Converts continuous IC50 values to categorical labels:
    - binary: active / inactive
    - ternary: active / intermediate / inactive

    Also computes pIC50 = -log10(IC50 in M).

    Example: drugflow data label -i curated.csv -o labeled.csv --mode binary
    """
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.bioactivity import label_activity
    from drugflow.phase1.data.writers import write_file

    click.echo(f"Loading {input_path}...")
    dataset = load_file(path=input_path)
    click.echo(f"  Loaded {len(dataset)} records")

    click.echo(f"Labeling ({mode}, active<{active_threshold} nM)...")
    labeled = label_activity(
        dataset,
        mode=mode,
        active_threshold=active_threshold,
        inactive_threshold=inactive_threshold if mode == "ternary" else None,
        value_key=value_key,
    )

    # Count classes
    valid = labeled.records
    n_active = sum(1 for r in valid if r.metadata.get("activity_class") == 1)
    n_inactive = sum(1 for r in valid
                     if r.metadata.get("activity_class") in (0, -1)
                     and r.metadata.get("activity_label") == "inactive")
    n_intermediate = sum(1 for r in valid
                         if r.metadata.get("activity_label") == "intermediate")

    click.echo(f"  Active:       {n_active}")
    if mode == "ternary":
        click.echo(f"  Intermediate: {n_intermediate}")
    click.echo(f"  Inactive:     {n_inactive}")

    write_file(labeled, output)
    click.echo(f"  Saved to {output}")


@data.command("fetch-and-curate")
@click.option("--target", "-t", required=True,
              help="ChEMBL target ID.")
@click.option("--activity-types", default="IC50",
              help="Comma-separated activity types.")
@click.option("--max-results", type=int, default=10000)
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output curated CSV.")
@click.option("--save-raw", type=click.Path(), default=None,
              help="Save raw data before curation.")
@click.option("--include-less-than", is_flag=True, default=False)
@click.option("--remove-outliers", is_flag=True, default=False)
@click.option("--label-mode", default="binary",
              type=click.Choice(["binary", "ternary"]),
              help="Activity labeling mode.")
@click.option("--active-threshold", type=float, default=1000.0,
              help="Active threshold in nM.")
@click.option("--inactive-threshold", type=float, default=10000.0,
              help="Inactive threshold in nM (ternary only).")
@click.pass_context
def fetch_and_curate(ctx, target, activity_types, max_results, output,
                     save_raw, include_less_than, remove_outliers,
                     label_mode, active_threshold, inactive_threshold):
    """Full pipeline: fetch from ChEMBL, curate, and label.

    Combines fetch-bioactivity, curate, and label into one command.

    Example: drugflow data fetch-and-curate -t CHEMBL4860 -o bcl2_ready.csv
    """
    from drugflow.phase1.data.bioactivity import fetch_and_curate as _fac
    from drugflow.phase1.data.writers import write_file

    types_list = [t.strip() for t in activity_types.split(",")]
    allowed = {"=", "<", "<="} if include_less_than else {"="}

    click.echo("=" * 60)
    click.echo("DrugFlow Fetch & Curate Pipeline")
    click.echo("=" * 60)

    click.echo(f"\n[1/3] Fetching from ChEMBL (target={target})...")
    click.echo(f"  Types: {types_list}, max: {max_results}")

    dataset, stats = _fac(
        target_id=target,
        activity_types=types_list,
        max_results=max_results,
        save_raw=save_raw,
        allowed_relations=allowed,
        remove_outliers=remove_outliers,
        label_mode=label_mode,
        active_threshold=active_threshold,
        inactive_threshold=inactive_threshold if label_mode == "ternary" else None,
    )

    click.echo(f"\n[2/3] Curation complete:")
    click.echo(f"  Fetched:   {stats.input_count}")
    click.echo(f"  Curated:   {stats.output_count}")
    click.echo(f"  Merged:    {stats.duplicates_merged} duplicates")
    if remove_outliers:
        click.echo(f"  Outliers:  {stats.outliers_removed} removed")

    valid = dataset.records
    n_active = sum(1 for r in valid if r.metadata.get("activity_class") == 1)
    n_inactive = sum(1 for r in valid
                     if r.metadata.get("activity_label") == "inactive")
    click.echo(f"  Active:    {n_active}")
    click.echo(f"  Inactive:  {n_inactive}")

    click.echo(f"\n[3/3] Saving to {output}...")
    write_file(dataset, output)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Done! {stats.output_count} curated molecules saved to {output}")
    click.echo(f"{'=' * 60}")
