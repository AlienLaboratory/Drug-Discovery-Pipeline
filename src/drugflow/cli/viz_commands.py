"""CLI commands for visualization."""

import click


@click.group()
def viz():
    """Visualization commands for molecular data."""
    pass


@viz.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output image path (PNG/SVG).")
@click.option("--grid", is_flag=True, default=False,
              help="Render as grid image.")
@click.option("--cols", type=int, default=5, help="Grid columns.")
@click.option("--max-mols", type=int, default=20,
              help="Max molecules to draw.")
@click.option("--label", default=None,
              help="Property name to use as label.")
@click.pass_context
def structures(ctx, input_path, output, grid, cols, max_mols, label):
    """Render 2D structure images."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.visualization.structure import draw_molecule, draw_molecule_grid

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    if label:
        from drugflow.phase1.analysis.properties import compute_properties_dataset
        dataset = compute_properties_dataset(dataset)

    if grid or len(dataset.valid_records) > 1:
        result = draw_molecule_grid(
            dataset, output, mols_per_page=max_mols,
            cols=cols, property_label=label,
        )
        click.echo(f"Saved molecule grid to {result}")
    else:
        valid = dataset.valid_records
        if valid:
            draw_molecule(valid[0].mol, output_path=output)
            click.echo(f"Saved molecule image to {output}")
        else:
            click.echo("No valid molecules to draw.")


@viz.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output plot path.")
@click.option("--properties", "-p", required=True,
              help="Comma-separated property names.")
@click.option("--plot-type", default="histogram",
              type=click.Choice(["histogram", "violin", "box", "kde"]))
@click.option("--cols", type=int, default=3,
              help="Subplot grid columns.")
@click.pass_context
def distributions(ctx, input_path, output, properties, plot_type, cols):
    """Plot property distributions."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase1.visualization.distributions import plot_property_distributions

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)
    dataset = compute_properties_dataset(dataset)

    prop_names = [p.strip() for p in properties.split(",")]
    result = plot_property_distributions(
        dataset, prop_names, output,
        plot_type=plot_type, ncols=cols,
    )
    click.echo(f"Saved distribution plot to {result}")


@viz.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output plot path.")
@click.option("--x", required=True, help="X-axis property.")
@click.option("--y", required=True, help="Y-axis property.")
@click.option("--color", default=None, help="Color-by property.")
@click.pass_context
def scatter(ctx, input_path, output, x, y, color):
    """Scatter plot of two properties."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase1.visualization.distributions import plot_property_scatter

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)
    dataset = compute_properties_dataset(dataset)

    result = plot_property_scatter(
        dataset, x, y, output, color_property=color,
    )
    click.echo(f"Saved scatter plot to {result}")


@viz.command("chemical-space")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output plot path.")
@click.option("--method", default="pca",
              type=click.Choice(["pca", "tsne"]))
@click.option("--fp-type", default="morgan",
              help="Fingerprint type.")
@click.option("--color", default=None, help="Color-by property.")
@click.option("--perplexity", type=float, default=30.0,
              help="t-SNE perplexity.")
@click.pass_context
def chemical_space(ctx, input_path, output, method, fp_type, color, perplexity):
    """Chemical space visualization via dimensionality reduction."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase1.analysis.fingerprints import compute_fingerprints_dataset
    from drugflow.phase1.visualization.chemical_space import (
        plot_chemical_space_pca,
        plot_chemical_space_tsne,
    )

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)

    if color:
        dataset = compute_properties_dataset(dataset)

    # Compute fingerprints
    fp_name = f"{fp_type}_r2_2048" if fp_type == "morgan" else fp_type
    fp_config = {fp_name: {"type": fp_type, "radius": 2, "nbits": 2048}}
    dataset = compute_fingerprints_dataset(dataset, fp_types=fp_config)

    if method == "pca":
        result = plot_chemical_space_pca(
            dataset, fp_type=fp_name, output_path=output,
            color_property=color,
        )
    else:
        result = plot_chemical_space_tsne(
            dataset, fp_type=fp_name, output_path=output,
            color_property=color, perplexity=perplexity,
        )

    click.echo(f"Saved chemical space plot to {result}")


@viz.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input file.")
@click.option("--output", "-o", required=True,
              type=click.Path(), help="Output plot path.")
@click.option("--properties", "-p", default=None,
              help="Comma-separated property names (all numeric if omitted).")
@click.pass_context
def correlation(ctx, input_path, output, properties):
    """Correlation heatmap of properties."""
    from drugflow.phase1.data.loaders import load_file
    from drugflow.phase1.data.validators import validate_dataset
    from drugflow.phase1.analysis.properties import compute_properties_dataset
    from drugflow.phase1.visualization.distributions import plot_property_correlation_matrix

    dataset = load_file(path=input_path)
    dataset = validate_dataset(dataset)
    dataset = compute_properties_dataset(dataset)

    prop_names = [p.strip() for p in properties.split(",")] if properties else None
    result = plot_property_correlation_matrix(
        dataset, property_names=prop_names, output_path=output,
    )
    click.echo(f"Saved correlation matrix to {result}")
