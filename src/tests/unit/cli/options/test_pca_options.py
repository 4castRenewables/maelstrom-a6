import a6.cli.main as main
import a6.cli.options as _options
import click
import pytest


@main.cli.command("test-pca-command")
@_options.pca.N_COMPONENTS
@_options.pca.N_COMPONENTS_START
@_options.pca.N_COMPONENTS_END
def pca_command(
    n_components: int,
    n_components_start: int,
    n_components_end: _options.cluster.MinClusterSizeEnd,
):
    click.echo(n_components)
    click.echo(n_components_start)
    click.echo(f"{n_components_end}")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["--n-components", "2", "--n-components-start", "3"],
            ["2", "3", "None"],
        ),
        (
            [
                "--n-components",
                "2",
                "--n-components-start",
                "3",
                "--n-components-end",
                "4",
            ],
            ["2", "3", "4"],
        ),
    ],
)
def test_pca_options(runner, args, expected):
    args = ["test-pca-command", *args]

    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0
    assert all(output in result.output for output in expected)
