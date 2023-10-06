import pathlib

import numpy as np
import pandas as pd
import pytest

import a6.datasets.era5 as era5
import a6.testing as testing

FILE_PATH = pathlib.Path(__file__).parent
# DATA_DIR = FILE_PATH / "../../data"
DATA_DIR = pathlib.Path("/home/fabian/data/era5")


class TestEra5:
    @pytest.mark.parametrize(
        ("path", "pattern", "slice_time_dimension", "expected"),
        [
            (
                DATA_DIR / "1979/01/1979010112_ml.grb",
                "should be ignored",
                False,
                # The data only contain a single time stamp at 12:00
                pd.date_range(
                    "1979-01-01T12:00", "1979-01-01T12:00", freq="1h"
                ),
            ),
            # Test case: expected 1978123112_ml.grb and 1979123112_ml.grb
            # to be read.
            (
                DATA_DIR,
                "**/197*_ml.grb",
                False,
                pd.date_range(
                    "1979-01-01T12:00", "1979-01-01T12:00", freq="1h"
                ).union(
                    pd.date_range(
                        "1979-12-31T12:00", "1979-12-31T12:00", freq="1h"
                    )
                ),
            ),
            # Test case: expected 1978123112_ml.grb to be read.
            (
                DATA_DIR,
                "**/1979010112_ml.grb",
                False,
                pd.date_range(
                    "1979-01-01T12:00", "1979-01-01T12:00", freq="1h"
                ),
            ),
        ],
    )
    def test_to_xarray(self, path, pattern, slice_time_dimension, expected):
        with testing.expect_raise_if_exception(expected):
            dataset = era5.Era5(
                path,
                pattern=pattern,
                slice_time_dimension=slice_time_dimension,
                parallel_loading=False,
            )
            result = dataset.to_xarray()
            result_dates = (
                result["time"].values.astype("datetime64[m]").tolist()
            )
            expected_dates = (
                expected.to_numpy().astype("datetime64[m]").tolist()
            )

            # Resulting dates might be a signle value instead of a list
            if not isinstance(result_dates, list):
                [expected_dates] = expected_dates

            np.testing.assert_equal(result_dates, expected_dates)
