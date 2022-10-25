import pathlib

import click.testing
import pytest

_FILE_PATH = pathlib.Path(__file__).parent


@pytest.fixture()
def runner() -> click.testing.CliRunner:
    return click.testing.CliRunner()


@pytest.fixture()
def config_path() -> pathlib.Path:
    return _FILE_PATH / "../../resources/test-config.yaml"
