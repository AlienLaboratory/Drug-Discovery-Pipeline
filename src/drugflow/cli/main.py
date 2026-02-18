"""Root CLI group for DrugFlow."""

import click

from drugflow import __version__


@click.group()
@click.option("--config", type=click.Path(exists=True), default=None,
              help="Path to YAML configuration file.")
@click.option("--output-dir", type=click.Path(), default="./output",
              help="Output directory for results.")
@click.option("--log-level", type=click.Choice(
    ["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO")
@click.option("--quiet", is_flag=True, help="Suppress progress bars.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.version_option(version=__version__, prog_name="drugflow")
@click.pass_context
def cli(ctx, config, output_dir, log_level, quiet, verbose):
    """DrugFlow - Drug Discovery Pipeline

    A modular command-line tool for computational drug discovery.
    Load molecular data, compute properties, apply filters,
    generate visualizations, and run analysis pipelines.
    """
    ctx.ensure_object(dict)
    if verbose:
        log_level = "DEBUG"
    ctx.obj["log_level"] = log_level
    ctx.obj["quiet"] = quiet
    ctx.obj["output_dir"] = output_dir
    ctx.obj["config"] = config

    from drugflow.core.logging import setup_logging
    setup_logging(level=log_level)


# Register command groups
from drugflow.cli.data_commands import data  # noqa: E402
from drugflow.cli.analyze_commands import analyze  # noqa: E402
from drugflow.cli.viz_commands import viz  # noqa: E402
from drugflow.cli.pipeline_commands import pipeline  # noqa: E402
from drugflow.cli.screen_commands import screen  # noqa: E402
from drugflow.cli.model_commands import model  # noqa: E402
from drugflow.cli.generate_commands import generate  # noqa: E402
from drugflow.cli.benchmark_commands import benchmark  # noqa: E402
from drugflow.cli.dock_commands import dock  # noqa: E402
from drugflow.cli.workflow_commands import workflow  # noqa: E402
from drugflow.cli.admet_commands import admet  # noqa: E402

cli.add_command(data)
cli.add_command(analyze)
cli.add_command(viz)
cli.add_command(pipeline)
cli.add_command(screen)
cli.add_command(model)
cli.add_command(generate)
cli.add_command(benchmark)
cli.add_command(dock)
cli.add_command(workflow)
cli.add_command(admet)


def main():
    cli()


if __name__ == "__main__":
    main()
