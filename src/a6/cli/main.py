import dataclasses
import pathlib
import sys
import typing as t

import click

Level = t.Optional[int]

DRY_RUN_KWARGS = {
    "color": "yellow",
    "bold": True,
}

WEATHER_DATA = click.option(
    "--weather-data",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Local or remote path to the weather data",
)


@dataclasses.dataclass(frozen=True)
class Options:
    """Options of the main CLI entry point."""

    dry_run: bool

    def __post_init__(self):
        if self.dry_run:
            click.secho(
                "INFO: Dry running, application code will not be executed",
                **DRY_RUN_KWARGS
            )

    def exit_if_dry_run(self) -> None:
        """Exit if user specified a dry run."""
        if self.dry_run:
            click.secho("INFO: Exiting dry run", **DRY_RUN_KWARGS)
            sys.exit(0)


GROUP_OPTIONS = click.make_pass_decorator(Options)

LEVEL = click.option(
    "-l",
    "--level",
    type=int,
    required=False,
    default=None,
    help="Level to select from the weather data",
)

LOG_TO_MANTIK = click.option(
    "--log-to-mantik",
    is_flag=True,
    help="Log to the Mantik platform (via MLflow)",
)


@click.group()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run the command without executing the application code.",
)
@click.pass_context
def cli(ctx, dry_run):
    """A6 CLI."""
    ctx.obj = Options(dry_run=dry_run)
