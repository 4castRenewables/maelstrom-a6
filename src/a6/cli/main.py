import logging

import a6.cli.options as options
import a6.utils as utils
import click


@click.group()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run the command without executing the application code.",
)
@click.option(
    "--log-to-mantik",
    type=click.BOOL,
    default=False,
    required=False,
    show_default=True,
    help="Log to the Mantik platform (via MLflow)",
)
@click.pass_context
def cli(ctx, dry_run: bool, log_to_mantik: bool):
    """A6 CLI."""
    utils.log_to_stdout(logging.DEBUG)
    ctx.obj = options.main.Options(dry_run=dry_run, log_to_mantik=log_to_mantik)
