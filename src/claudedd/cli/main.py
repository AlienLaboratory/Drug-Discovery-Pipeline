"""Root CLI group for ClaudeDD."""

import click

from claudedd import __version__


@click.group()
@click.option("--config", type=click.Path(exists=True), default=None,
              help="Path to YAML configuration file.")
@click.option("--output-dir", type=click.Path(), default="./output",
              help="Output directory for results.")
@click.option("--log-level", type=click.Choice(
    ["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO")
@click.option("--quiet", is_flag=True, help="Suppress progress bars.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.version_option(version=__version__, prog_name="claudedd")
@click.pass_context
def cli(ctx, config, output_dir, log_level, quiet, verbose):
    """ClaudeDD - Drug Discovery Pipeline

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

    from claudedd.core.logging import setup_logging
    setup_logging(level=log_level)


# Register command groups
from claudedd.cli.data_commands import data  # noqa: E402
from claudedd.cli.analyze_commands import analyze  # noqa: E402
from claudedd.cli.viz_commands import viz  # noqa: E402
from claudedd.cli.pipeline_commands import pipeline  # noqa: E402
from claudedd.cli.screen_commands import screen  # noqa: E402
from claudedd.cli.model_commands import model  # noqa: E402
from claudedd.cli.generate_commands import generate  # noqa: E402
from claudedd.cli.benchmark_commands import benchmark  # noqa: E402
from claudedd.cli.dock_commands import dock  # noqa: E402
from claudedd.cli.workflow_commands import workflow  # noqa: E402

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


def main():
    cli()


if __name__ == "__main__":
    main()
