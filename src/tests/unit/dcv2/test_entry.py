import pytest
import xarray as xr

import a6.datasets as datasets
import a6.dcv2.entry as entry
import a6.utils as utils


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


def test_train_dcv2(tmp_path):
    # Train first epoc
    raw_args_1 = [
        "--use-cpu",
        "--epochs",
        "1",
        "--dump-path",
        tmp_path.as_posix(),
    ]
    entry.train_dcv2(raw_args_1)

    # Train second epoch to restore from dump path
    raw_args_2 = raw_args_1 + ["--epochs", "2"]
    entry.train_dcv2(raw_args_2)


def test_train_dcv2_with_era5(tmp_path, mock_select_dwd_area, era5_path):
    # TODO:
    # Allow multiple levels for training
    #
    # Train first epoc without cutting DWD area
    raw_args_1 = [
        "--use-cpu",
        "--epochs",
        "1",
        "--data-path",
        era5_path.as_posix(),
        "--no-parallel-loading",
        "--pattern",
        "**/*.nc",
        "--level",
        "500",
        "--dump-path",
        tmp_path.as_posix(),
    ]
    entry.train_dcv2(raw_args_1)

    # Train second epoch to restore from dump path and cut DWD area
    raw_args_2 = raw_args_1 + ["--epochs", "2", "--select-dwd-area"]
    entry.train_dcv2(raw_args_2)
