import pathlib

import numpy as np
import pandas as pd
import pytest

import a6.datasets.ecmwf_ifs_hres as ecmwf_ifs_hres
import a6.testing as testing

FILE_PATH = pathlib.Path(__file__).parent
DATA_DIR = FILE_PATH / "../../data"


class TestEcmwfIfsHresDataset:
    @pytest.mark.parametrize(
        ("path", "pattern", "slice_time_dimension", "expected"),
        [
            (
                DATA_DIR / "ml_20190101_00.nc",
                "should be ignored",
                False,
                pd.date_range(
                    "2019-01-01T00:00", "2019-01-03T00:00", freq="1h"
                ),
            ),
            # Test case: expected ml_20190101_00.nc and ml_20190101_12.nc
            # to be read.
            (
                DATA_DIR,
                "ml_20190101_*.nc",
                True,
                pd.date_range(
                    "2019-01-01T00:00", "2019-01-01T23:00", freq="1h"
                ),
            ),
            # Test case: expected ml_20190101_12.nc to be read.
            (
                DATA_DIR,
                "ml_20190101_12.nc",
                True,
                pd.date_range(
                    "2019-01-01T12:00", "2019-01-01T23:00", freq="1h"
                ),
            ),
        ],
    )
    def test_to_xarray(self, path, pattern, slice_time_dimension, expected):
        with testing.expect_raise_if_exception(expected):
            dataset = ecmwf_ifs_hres.EcmwfIfsHres(
                path,
                pattern=pattern,
                slice_time_dimension=slice_time_dimension,
                parallel_loading=False,
            )
            result = dataset.to_xarray()
            result_dates = result["time"].values.astype("datetime64[m]")
            expected_dates = expected.to_numpy().astype("datetime64[m]")

            np.testing.assert_equal(result_dates, expected_dates)
