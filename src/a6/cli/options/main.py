import dataclasses
import sys

import click

_DRY_RUN_KWARGS = {
    "color": "yellow",
    "bold": True,
}


@dataclasses.dataclass(frozen=True)
class Options:
    """Options of the main CLI entry point."""

    dry_run: bool
    log_to_mantik: bool

    def __post_init__(self):
        if self.dry_run:
            click.secho(
                "INFO: Dry running, application code will not be executed",
                **_DRY_RUN_KWARGS
            )

    def exit_if_dry_run(self) -> None:
        """Exit if user specified a dry run."""
        if self.dry_run:
            click.secho("INFO: Exiting dry run", **_DRY_RUN_KWARGS)
            sys.exit(0)


PASS_OPTIONS = click.make_pass_decorator(Options)
