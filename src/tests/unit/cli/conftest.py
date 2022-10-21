import click.testing
import pytest


@pytest.fixture()
def runner() -> click.testing.CliRunner:
    return click.testing.CliRunner()
