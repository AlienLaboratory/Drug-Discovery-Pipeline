"""CLI commands for benchmarking generation strategies."""

import click


@click.group()
def benchmark():
    """Benchmark and compare molecular generation strategies."""
    pass


@benchmark.command("run")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input seed molecules file.")
@click.option("--reference", "-r", "reference_path", default=None,
              type=click.Path(exists=True), help="Reference molecules for novelty.")
@click.option("--strategies", "-s", default="brics,mutate",
              help="Comma-separated list of strategies: brics,scaffold,mutate")
@click.option("-n", "n_molecules", default=50, type=int,
              help="Target molecules per strategy.")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output directory or CSV file for results.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.pass_context
def run_benchmark_cmd(ctx, input_path, reference_path, strategies,
                      n_molecules, output_path, seed):
    """Run generation strategies and compare metrics."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase3.generation.brics_enum import generate_brics
    from drugflow.phase3.generation.mutations import generate_mutations
    from drugflow.phase3.generation.scaffold_decoration import generate_from_scaffold
    from drugflow.phase3.benchmarking.benchmark_runner import (
        BenchmarkStrategy, run_benchmark,
    )
    from drugflow.phase3.benchmarking.comparison import (
        compare_results, export_comparison,
    )

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    reference = None
    if reference_path:
        reference = load_file(path=reference_path)
        reference = validate_dataset(reference)

    # Build strategy list
    strategy_map = {
        "brics": BenchmarkStrategy(
            name="brics",
            generator_fn=generate_brics,
            params={"seed": seed},
        ),
        "scaffold": BenchmarkStrategy(
            name="scaffold",
            generator_fn=lambda ds, n_molecules, seed=42: generate_from_scaffold(
                ds.valid_records[0].mol if ds.valid_records else None,
                n_molecules=n_molecules, seed=seed,
            ),
            params={"seed": seed},
        ),
        "mutate": BenchmarkStrategy(
            name="mutate",
            generator_fn=generate_mutations,
            params={"seed": seed},
        ),
    }

    strategy_names = [s.strip() for s in strategies.split(",")]
    strategy_list = []
    for name in strategy_names:
        if name in strategy_map:
            strategy_list.append(strategy_map[name])
        else:
            click.echo(f"Warning: Unknown strategy '{name}', skipping.")

    if not strategy_list:
        click.echo("Error: No valid strategies specified.", err=True)
        return

    click.echo(
        f"Benchmarking {len(strategy_list)} strategies on "
        f"{len(dataset.valid_records)} molecules..."
    )

    results = run_benchmark(
        strategy_list, dataset, n_molecules=n_molecules,
        reference_dataset=reference,
    )

    # Display comparison table
    df = compare_results(results)
    click.echo("\n" + str(df))

    if output_path:
        export_comparison(results, output_path)
        click.echo(f"\nResults saved to {output_path}")


@benchmark.command("compare")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input benchmark results CSV.")
@click.option("--output", "-o", "output_path", type=click.Path(),
              help="Output comparison file.")
@click.option("--rank-by", default="internal_diversity",
              help="Metric to rank strategies by.")
@click.pass_context
def compare_cmd(ctx, input_path, output_path, rank_by):
    """Compare benchmark results from a previous run."""
    import pandas as pd

    df = pd.read_csv(input_path, index_col=0)
    click.echo("Benchmark Results:")
    click.echo(str(df))

    if rank_by in df.columns:
        ranked = df.sort_values(rank_by, ascending=False)
        click.echo(f"\nRanked by {rank_by}:")
        for idx, row in ranked.iterrows():
            click.echo(f"  {idx}: {row[rank_by]:.4f}")

    if output_path:
        df.to_csv(output_path)
        click.echo(f"Saved to {output_path}")
