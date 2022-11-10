import sklearn.cluster as cluster

import a6.cli.main as main
import a6.cli.options as _options


@main.cli.command("test-config-command")
@_options.config.CONFIG
def pca_command(
    config: _options.config.Config,
):
    assert config.model == cluster.KMeans
    assert config.parameters == {"test_parameter": 1}


def test_config_options(runner, config_path):
    args = ["test-config-command", "--config", config_path.as_posix()]

    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0
