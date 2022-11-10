import click
import pytest

import a6.cli.main as main
import a6.cli.options as _options


@main.cli.command("test-pca-command")
@_options.pca.N_COMPONENTS
@_options.pca.USE_VARIMAX
def pca_command(
    n_components: int,
    use_varimax: bool,
):
    click.echo(n_components)
    click.echo(use_varimax)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["--n-components", "2"],
            ["2", "False"],
        ),
        (
            ["--n-components", "2", "--use-varimax", "true"],
            ["2", "True"],
        ),
    ],
)
def test_pca_options(runner, args, expected):
    args = ["test-pca-command", *args]

    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0
    assert all(output in result.output for output in expected)
