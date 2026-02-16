"""CLI commands for molecular generation."""

import click


@click.group()
def generate():
    """De novo molecular generation: BRICS, scaffold, mutation, GA, active learning."""
    pass


@generate.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input seed molecules file.")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output file for generated molecules.")
@click.option("-n", "n_molecules", default=100, type=int,
              help="Number of molecules to generate.")
@click.option("--min-freq", default=2, type=int,
              help="Minimum fragment frequency for library.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.pass_context
def brics(ctx, input_path, output_path, n_molecules, min_freq, seed):
    """Generate molecules via BRICS fragment recombination."""
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase3.generation.brics_enum import generate_brics
    from claudedd.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    click.echo(f"BRICS generation from {len(dataset.valid_records)} seed molecules...")
    generated = generate_brics(
        dataset, n_molecules=n_molecules,
        min_fragment_frequency=min_freq, seed=seed,
    )

    click.echo(f"Generated: {len(generated)} molecules")
    for rec in generated.valid_records[:5]:
        click.echo(f"  {rec.source_id}: {rec.canonical_smiles[:60]}")
    if len(generated) > 5:
        click.echo(f"  ... and {len(generated) - 5} more")

    if output_path:
        write_file(generated, output_path)
        click.echo(f"Saved to {output_path}")


@generate.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input seed molecules file.")
@click.option("--scaffold", "scaffold_smi", default=None,
              help="Scaffold SMILES to decorate. Uses first molecule if not specified.")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output file for generated molecules.")
@click.option("-n", "n_molecules", default=100, type=int,
              help="Number of molecules to generate.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.pass_context
def scaffold(ctx, input_path, scaffold_smi, output_path, n_molecules, seed):
    """Generate molecules via scaffold decoration with R-groups."""
    from rdkit import Chem
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase3.generation.scaffold_decoration import generate_from_scaffold
    from claudedd.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    if scaffold_smi:
        seed_mol = Chem.MolFromSmiles(scaffold_smi)
        if seed_mol is None:
            click.echo(f"Error: Invalid scaffold SMILES: {scaffold_smi}", err=True)
            return
    else:
        valid = dataset.valid_records
        if not valid:
            click.echo("Error: No valid molecules in input", err=True)
            return
        seed_mol = valid[0].mol

    click.echo(f"Scaffold decoration generating {n_molecules} molecules...")
    generated = generate_from_scaffold(seed_mol, n_molecules=n_molecules, seed=seed)

    click.echo(f"Generated: {len(generated)} molecules")
    for rec in generated.valid_records[:5]:
        click.echo(f"  {rec.source_id}: {rec.canonical_smiles[:60]}")

    if output_path:
        write_file(generated, output_path)
        click.echo(f"Saved to {output_path}")


@generate.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input seed molecules file.")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output file for generated molecules.")
@click.option("-n", "n_molecules", default=100, type=int,
              help="Number of molecules to generate.")
@click.option("--n-mutations", default=1, type=int,
              help="Number of mutations per molecule.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.pass_context
def mutate(ctx, input_path, output_path, n_molecules, n_mutations, seed):
    """Generate molecules via random chemical mutations."""
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase3.generation.mutations import generate_mutations
    from claudedd.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    click.echo(f"Mutation generation from {len(dataset.valid_records)} seed molecules...")
    generated = generate_mutations(
        dataset, n_molecules=n_molecules,
        n_mutations_per_mol=n_mutations, seed=seed,
    )

    click.echo(f"Generated: {len(generated)} molecules")
    for rec in generated.valid_records[:5]:
        click.echo(f"  {rec.source_id}: {rec.canonical_smiles[:60]}")

    if output_path:
        write_file(generated, output_path)
        click.echo(f"Saved to {output_path}")


@generate.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input seed molecules file.")
@click.option("--model", "-m", "model_path", required=True,
              type=click.Path(exists=True), help="Trained QSAR model (joblib).")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output file for generated molecules.")
@click.option("-n", "n_molecules", default=100, type=int,
              help="GA population size.")
@click.option("--generations", default=50, type=int,
              help="Number of GA generations.")
@click.option("--mutation-rate", default=0.3, type=float,
              help="Mutation probability per generation.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.pass_context
def ga(ctx, input_path, model_path, output_path, n_molecules, generations,
       mutation_rate, seed):
    """Run genetic algorithm guided by QSAR model."""
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase1.analysis.descriptors import compute_descriptors_dataset
    from claudedd.phase2.qsar.persistence import load_model
    from claudedd.phase3.generation.genetic_algorithm import (
        GeneticAlgorithmConfig, qsar_guided_ga,
    )
    from claudedd.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)
    dataset = compute_descriptors_dataset(dataset)

    model = load_model(model_path)
    click.echo(f"Loaded model: {model.model_type} ({model.task})")

    config = GeneticAlgorithmConfig(
        population_size=n_molecules,
        n_generations=generations,
        mutation_rate=mutation_rate,
        seed=seed,
    )

    click.echo(f"Running GA: pop={n_molecules}, gen={generations}...")
    generated = qsar_guided_ga(dataset, model, config=config)

    click.echo(f"Generated: {len(generated)} unique molecules")
    for rec in generated.valid_records[:5]:
        fitness = rec.properties.get("ga_fitness", "N/A")
        click.echo(f"  {rec.source_id}: {rec.canonical_smiles[:50]} (fitness={fitness:.4f})")

    if output_path:
        write_file(generated, output_path)
        click.echo(f"Saved to {output_path}")


@generate.command("active-pick")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Pool of candidate molecules.")
@click.option("--model", "-m", "model_path", required=True,
              type=click.Path(exists=True), help="Trained QSAR model (joblib).")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output file for selected batch.")
@click.option("--strategy", default="ucb",
              type=click.Choice(["greedy", "uncertainty", "ucb", "diversity", "balanced"]),
              help="Acquisition strategy.")
@click.option("--batch-size", default=20, type=int,
              help="Number of molecules to select.")
@click.option("--kappa", default=1.0, type=float,
              help="UCB exploration parameter (higher = more exploration).")
@click.pass_context
def active_pick(ctx, input_path, model_path, output_path, strategy,
                batch_size, kappa):
    """Select molecules for testing via active learning."""
    from claudedd.phase1.data.loaders import load_file
    from claudedd.phase1.data.validators import validate_dataset
    from claudedd.phase1.analysis.descriptors import compute_descriptors_dataset
    from claudedd.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from claudedd.phase2.qsar.persistence import load_model
    from claudedd.phase1.data.writers import write_file

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)
    dataset = compute_descriptors_dataset(dataset)
    dataset = compute_fingerprints_dataset(dataset)

    model = load_model(model_path)
    click.echo(f"Active learning: strategy={strategy}, batch_size={batch_size}")

    if strategy == "greedy":
        from claudedd.phase3.active_learning.acquisition import greedy_acquisition
        selected = greedy_acquisition(dataset, model, batch_size)
    elif strategy == "uncertainty":
        from claudedd.phase3.active_learning.acquisition import uncertainty_acquisition
        selected = uncertainty_acquisition(dataset, model, batch_size)
    elif strategy == "ucb":
        from claudedd.phase3.active_learning.acquisition import ucb_acquisition
        selected = ucb_acquisition(dataset, model, batch_size, kappa=kappa)
    elif strategy == "diversity":
        from claudedd.phase3.active_learning.acquisition import diversity_acquisition
        selected = diversity_acquisition(dataset, batch_size)
    elif strategy == "balanced":
        from claudedd.phase3.active_learning.acquisition import balanced_acquisition
        selected = balanced_acquisition(dataset, model, batch_size)

    click.echo(f"Selected: {len(selected)} molecules")
    for rec in selected.valid_records[:5]:
        click.echo(f"  {rec.source_id or rec.record_id}: {rec.canonical_smiles[:60]}")

    if output_path:
        write_file(selected, output_path)
        click.echo(f"Saved to {output_path}")
