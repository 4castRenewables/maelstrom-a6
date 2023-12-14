import pathlib

import pytest
import xarray as xr

import a6.datasets as datasets
import a6.dcv2.entry as entry
import a6.testing as testing
import a6.utils as utils
import mlflow


@utils.functional.make_functional
def dummy_method(ds: xr.Dataset, *args, **kwargs) -> xr.Dataset:
    return ds


@pytest.fixture
def mock_select_dwd_area(monkeypatch) -> None:
    monkeypatch.setattr(
        datasets.methods.select,
        "select_dwd_area",
        dummy_method,
    )


def _create_stdout_stderr_file(path: pathlib.Path) -> tuple[str, str]:
    stdout = path / "stdout"
    stderr = path / "stderr"
    with open(stdout, "w+") as f:
        f.write("test-stdout")
    with open(stderr, "w+") as f:
        f.write("test-stderr")
    return stdout.as_posix(), stderr.as_posix()


def test_train_dcv2(tmp_path):
    stdout, stderr = _create_stdout_stderr_file(tmp_path)

    # Train first epoc
    raw_args_1 = [
        "--enable-tracking",
        "--use-cpu",
        "--epochs",
        "1",
        "--dump-path",
        tmp_path.as_posix(),
    ]

    with testing.env.env_vars_set(
        {
            "MLFLOW_TRACKING_URI": (tmp_path / "mlruns").as_uri(),
            "SLURM_JOB_STDOUT": stdout,
            "SLURM_JOB_STDERR": stderr,
        }
    ):
        run = mlflow.start_run()
        mlflow.end_run()
        with testing.env.env_vars_set({"MLFLOW_RUN_ID": run.info.run_id}):
            entry.train_dcv2(raw_args_1)

            # Train second epoch to restore from dump path
            raw_args_2 = raw_args_1 + ["--epochs", "2"]
            entry.train_dcv2(raw_args_2)


@pytest.mark.parametrize("levels", [["500"], ["500", "950"]])
def test_train_dcv2_with_era5(
    tmp_path, mock_select_dwd_area, era5_path, levels
):
    stdout, stderr = _create_stdout_stderr_file(tmp_path)

    # Train first epoc without cutting DWD area
    raw_args_1 = [
        "--enable-tracking",
        "--use-cpu",
        "--epochs",
        "1",
        "--data-path",
        era5_path.as_posix(),
        "--no-parallel-loading",
        "--pattern",
        "**/*.nc",
        "--level",
        *levels,
        "--dump-path",
        tmp_path.as_posix(),
    ]
    with testing.env.env_vars_set(
        {
            "MLFLOW_TRACKING_URI": (tmp_path / "mlruns").as_uri(),
            "SLURM_JOB_STDOUT": stdout,
            "SLURM_JOB_STDERR": stderr,
        }
    ):
        run = mlflow.start_run()
        mlflow.end_run()
        with testing.env.env_vars_set({"MLFLOW_RUN_ID": run.info.run_id}):
            entry.train_dcv2(raw_args_1)

            # Train second epoch to restore from dump path and cut DWD area
            raw_args_2 = raw_args_1 + ["--epochs", "2", "--select-dwd-area"]
            entry.train_dcv2(raw_args_2)
