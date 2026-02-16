"""CLI commands for structure-based drug design and docking."""

import click


@click.group()
def dock():
    """Structure-based drug design: ligand prep, shape screening, interactions, docking."""
    pass


@dock.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input molecules file (CSV/SDF/SMI).")
@click.option("--output-dir", "-o", "output_dir", required=True,
              type=click.Path(), help="Output directory for prepared ligands.")
@click.option("--n-confs", default=20, type=int,
              help="Number of conformers to generate per molecule.")
@click.option("--force-field", "force_field", default="MMFF",
              type=click.Choice(["MMFF", "UFF"]),
              help="Force field for optimization.")
@click.option("--no-optimize", is_flag=True, help="Skip energy optimization.")
@click.pass_context
def prepare(ctx, input_path, output_dir, n_confs, force_field, no_optimize):
    """Prepare ligands: 3D conformers, optimization, export.

    Generates 3D conformers using ETKDGv3, optimizes with MMFF/UFF,
    prunes redundant conformers, and exports to SDF.
    """
    import os
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase4.ligand_prep.preparation import (
        prepare_ligand_dataset, export_ligand_sdf,
    )

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    click.echo(f"Preparing {len(dataset.valid_records)} ligands "
               f"(n_confs={n_confs}, ff={force_field}, optimize={not no_optimize})...")

    dataset = prepare_ligand_dataset(
        dataset, n_confs=n_confs,
        optimize=not no_optimize, force_field=force_field,
    )

    os.makedirs(output_dir, exist_ok=True)

    exported = 0
    for rec in dataset.valid_records:
        if rec.mol is not None and rec.mol.GetNumConformers() > 0:
            name = rec.source_id or rec.record_id
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            out_path = os.path.join(output_dir, f"{safe_name}_3d.sdf")
            try:
                export_ligand_sdf(rec.mol, out_path)
                exported += 1
            except Exception as e:
                click.echo(f"  Warning: Failed to export {name}: {e}", err=True)

    click.echo(f"Prepared and exported {exported} ligands to {output_dir}")

    # Summary stats
    for rec in dataset.valid_records[:5]:
        n_conf = rec.mol.GetNumConformers() if rec.mol else 0
        energy = rec.properties.get("lowest_conformer_energy", "N/A")
        if isinstance(energy, float):
            energy = f"{energy:.1f}"
        click.echo(f"  {rec.source_id}: {n_conf} conformers, energy={energy}")
    if len(dataset.valid_records) > 5:
        click.echo(f"  ... and {len(dataset.valid_records) - 5} more")


@dock.command("shape-screen")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Library molecules file.")
@click.option("--reference", "-r", "ref_path", required=True,
              type=click.Path(exists=True), help="Reference molecule file (SDF).")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output file for shape hits.")
@click.option("--threshold", default=0.3, type=float,
              help="Minimum shape similarity score.")
@click.option("--metric", default="tanimoto",
              type=click.Choice(["tanimoto", "protrusion", "combo"]),
              help="Shape similarity metric.")
@click.pass_context
def shape_screen(ctx, input_path, ref_path, output_path, threshold, metric):
    """Screen a library by 3D shape similarity to a reference.

    Uses RDKit's shape alignment and scoring to find molecules
    with similar 3D shape to a reference compound.
    """
    from rdkit import Chem
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase4.shape_screening.shape_screen import screen_by_shape
    from claudedd.phase1.data.writers import write_file

    # Load library
    library = load_file(path=input_path)
    library = validate_dataset(library)

    # Load reference
    ref_dataset = load_file(path=ref_path)
    ref_mols = [r.mol for r in ref_dataset.records if r.mol is not None]
    if not ref_mols:
        click.echo("Error: No valid reference molecule found.", err=True)
        return
    reference = ref_mols[0]

    click.echo(f"Shape screening: {len(library.valid_records)} molecules "
               f"vs reference (metric={metric}, threshold={threshold})...")

    hits = screen_by_shape(library, reference, threshold=threshold, metric=metric)

    click.echo(f"Hits: {len(hits.valid_records)} molecules above threshold {threshold}")
    for rec in hits.valid_records[:10]:
        score = rec.properties.get(f"shape_{metric}", "N/A")
        if isinstance(score, float):
            score = f"{score:.3f}"
        click.echo(f"  {rec.source_id}: score={score}")

    if output_path:
        write_file(hits, output_path)
        click.echo(f"Saved {len(hits.valid_records)} hits to {output_path}")


@dock.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Ligand file (SDF with 3D).")
@click.option("--protein", "-p", "protein_path", default=None,
              type=click.Path(exists=True), help="Protein PDB file (optional).")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output JSON file for interactions.")
@click.pass_context
def interactions(ctx, input_path, protein_path, output_path):
    """Analyze protein-ligand interactions.

    Detects hydrogen bonds, hydrophobic contacts, pi-stacking,
    and salt bridges. If no protein is provided, performs self-analysis
    (geometry measurements only).
    """
    import json
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase4.interactions.geometry import (
        compute_molecular_volume, compute_bounding_box,
    )

    dataset = load_file(path=input_path)
    valid = [r for r in dataset.records if r.mol is not None]
    if not valid:
        click.echo("Error: No valid molecules in input.", err=True)
        return

    results = {}
    ligand = valid[0]

    # Always compute geometry
    if ligand.mol.GetNumConformers() > 0:
        try:
            vol = compute_molecular_volume(ligand.mol)
            bb_min, bb_max = compute_bounding_box(ligand.mol)
            results["geometry"] = {
                "volume": round(vol, 2),
                "bounding_box_min": bb_min.tolist(),
                "bounding_box_max": bb_max.tolist(),
            }
            click.echo(f"Ligand: {ligand.source_id}")
            click.echo(f"  Volume: {vol:.2f} A^3")
            click.echo(f"  Bounding box: {bb_min.round(1).tolist()} to {bb_max.round(1).tolist()}")
        except Exception as e:
            click.echo(f"  Geometry error: {e}", err=True)

    # If protein provided, compute contacts
    if protein_path:
        from claudedd.phase4.docking.protein_prep import prepare_protein
        from claudedd.phase4.interactions.contacts import detect_all_contacts
        from claudedd.phase4.interactions.plif import compute_plif

        click.echo(f"Loading protein: {protein_path}")
        try:
            protein = prepare_protein(protein_path)
            contacts = detect_all_contacts(ligand.mol, protein)
            plif = compute_plif(ligand.mol, protein)

            results["contacts"] = {
                "hbonds": len(contacts["hbonds"]),
                "hydrophobic": len(contacts["hydrophobic"]),
                "pi_stacking": len(contacts["pi_stacking"]),
                "salt_bridges": len(contacts["salt_bridges"]),
            }
            results["plif"] = plif.tolist()

            click.echo(f"  H-bonds: {len(contacts['hbonds'])}")
            click.echo(f"  Hydrophobic: {len(contacts['hydrophobic'])}")
            click.echo(f"  Pi-stacking: {len(contacts['pi_stacking'])}")
            click.echo(f"  Salt bridges: {len(contacts['salt_bridges'])}")
        except Exception as e:
            click.echo(f"  Interaction analysis failed: {e}", err=True)
    else:
        click.echo("  (No protein provided — geometry-only analysis)")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"Saved interactions to {output_path}")


@dock.command("run")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Prepared ligands file.")
@click.option("--protein", "-p", "protein_path", required=True,
              type=click.Path(exists=True), help="Protein PDBQT file.")
@click.option("--center", required=True,
              help="Docking box center 'x,y,z' in Angstroms.")
@click.option("--size", default="20,20,20",
              help="Docking box size 'x,y,z' in Angstroms.")
@click.option("--exhaustiveness", default=8, type=int,
              help="Vina exhaustiveness parameter.")
@click.option("--n-modes", default=9, type=int,
              help="Number of binding modes to return.")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output file for docking results.")
@click.pass_context
def run(ctx, input_path, protein_path, center, size, exhaustiveness,
        n_modes, output_path):
    """Run molecular docking with AutoDock Vina.

    Requires: pip install vina meeko

    Docks ligands against a protein target within a defined search box.
    """
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase4.docking.grid import define_grid_from_coords
    from claudedd.phase4.docking.vina_wrapper import dock_dataset_vina
    from claudedd.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    # Parse center and size
    try:
        cx, cy, cz = [float(x.strip()) for x in center.split(",")]
        sx, sy, sz = [float(x.strip()) for x in size.split(",")]
    except ValueError:
        click.echo("Error: center and size must be 'x,y,z' format", err=True)
        return

    box = define_grid_from_coords((cx, cy, cz), (sx, sy, sz))
    click.echo(f"Docking {len(dataset.valid_records)} molecules...")
    click.echo(f"  Protein: {protein_path}")
    click.echo(f"  Box center: ({cx}, {cy}, {cz}), size: ({sx}, {sy}, {sz})")
    click.echo(f"  Exhaustiveness: {exhaustiveness}, modes: {n_modes}")

    try:
        dataset = dock_dataset_vina(
            dataset, protein_path, box,
            exhaustiveness=exhaustiveness, n_modes=n_modes,
        )
    except Exception as e:
        click.echo(f"Docking failed: {e}", err=True)
        return

    # Report results
    docked = [r for r in dataset.valid_records if "vina_score" in r.properties]
    click.echo(f"Successfully docked: {len(docked)} molecules")
    for rec in sorted(docked, key=lambda r: r.properties["vina_score"])[:10]:
        score = rec.properties["vina_score"]
        click.echo(f"  {rec.source_id}: {score:.2f} kcal/mol")

    if output_path:
        write_file(dataset, output_path)
        click.echo(f"Saved results to {output_path}")
