import typing as t

import click

Components = int
ComponentsEnd = t.Optional[int]

N_COMPONENTS = click.option(
    "--n-components",
    type=int,
    required=True,
    help="Number of components to use for PCA.",
)

USE_VARIMAX = click.option(
    "--use-varimax",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Whether to use VariMax rotation for the PCs.",
)
