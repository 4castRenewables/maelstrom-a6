import pathlib
from contextlib import nullcontext as doesnotraise

import a6.datasets.ecmwf_ifs_hres as ecmwf_ifs_hres
import numpy as np
import pandas as pd
import pytest

FILE_PATH = pathlib.Path(__file__).parent
DATA_DIR = FILE_PATH / "../../data"


class TestEcmwfIfsHresDataset:
    @pytest.mark.parametrize(
        ("paths", "overlapping", "expected"),
        [
            ([], False, ValueError()),
            (
                [DATA_DIR / "ml_20190101_00.nc"],
                False,
                pd.date_range(
                    "2019-01-01T00:00", "2019-01-03T00:00", freq="1h"
                ),
            ),
            (
                [
                    DATA_DIR / "ml_20190101_00.nc",
                    DATA_DIR / "ml_20190101_12.nc",
                ],
                True,
                pd.date_range(
                    "2019-01-01T00:00", "2019-01-01T23:00", freq="1h"
                ),
            ),
        ],
    )
    def test_as_xarray(self, paths, overlapping, expected):
        with pytest.raises(type(expected)) if isinstance(
            expected, Exception
        ) else doesnotraise():
            dataset = ecmwf_ifs_hres.EcmwfIfsHres(
                paths, overlapping=overlapping, parallel_loading=False
            )
            result = dataset.as_xarray()
            result_dates = result["time"].values.astype("datetime64[m]")
            expected_dates = expected.to_numpy().astype("datetime64[m]")

            np.testing.assert_equal(result_dates, expected_dates)
