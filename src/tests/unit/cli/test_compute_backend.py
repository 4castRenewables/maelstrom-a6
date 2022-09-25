import os

import a6.cli.compute_backend as compute_backend
import mantik.utils.mlflow as mlflow
import pytest


@pytest.mark.parametrize(
    (
        "set_experiment_id_env_var",
        "cli_args",
        "expected_entry_point",
        "expected_mlflow_parameters",
        "expected_experiment_id",
        "expected_backend_config",
    ),
    [
        (
            False,
            [
                "--experiment-id",
                "1",
                "--entry-point",
                "main",
                "--param",
                "key1=1",
                "--param",
                "key2=value",
                "--param",
                "key3=False",
                "--param",
                'key4="double quoted string"',
                "--param",
                "key5='single quoted string'",
                "--param",
                'path="/is/a/path"',
                "--backend-config",
                "some-name.json",
            ],
            "main",
            [
                ("key1", 1),
                ("key2", "value"),
                ("key3", False),
                ("key4", "double quoted string"),
                ("key5", "single quoted string"),
                ("path", "/is/a/path"),
            ],
            1,
            "some-name.json",
        ),
        # Test case: experiment ID and backend config not provided
        (
            True,
            [
                "--entry-point",
                "main",
                "--param",
                "test=1",
            ],
            "main",
            [("test", 1)],
            None,
            "unicore-config.json",
        ),
    ],
)
def test_compute_backend_parser(
    set_experiment_id_env_var,
    cli_args,
    expected_entry_point,
    expected_mlflow_parameters,
    expected_experiment_id,
    expected_backend_config,
):
    if set_experiment_id_env_var:
        os.environ[mlflow.EXPERIMENT_ID_ENV_VAR] = "1"
    else:
        os.environ.pop(mlflow.EXPERIMENT_ID_ENV_VAR, None)

    parser = compute_backend.create_parser()
    args = parser.parse_args(cli_args)

    assert args.entry_point == expected_entry_point
    assert args.mlflow_parameters == expected_mlflow_parameters
    assert args.experiment_id == expected_experiment_id
    assert args.backend_config == expected_backend_config
