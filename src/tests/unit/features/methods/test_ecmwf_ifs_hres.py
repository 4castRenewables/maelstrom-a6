import pathlib

import lifetimes.features.methods.ecmwf_ifs_hres as ecmwf_ifs_hres
import numpy as np
import pytest
import xarray as xr

FILE_DIR = pathlib.Path(__file__).parent


@pytest.fixture()
def pl_ds() -> xr.Dataset:
    return xr.open_dataset(FILE_DIR / "../../../data/pl_20201201_00.nc")


@pytest.mark.parametrize("level", [500, [500, 1000]])
def test_select_level(pl_ds, level):
    result = ecmwf_ifs_hres.select_level(pl_ds, level=level)

    assert result["level"].values.tolist() == level


def test_select_level_and_calculate_daily_mean(pl_ds):
    expected = np.array(
        [
            [
                [245.4356, 245.53772],
                [245.34428, 245.38219],
                [245.47972, 245.4964],
            ],
            [
                [235.93938, 235.96883],
                [235.544, 235.57054],
                [235.26147, 235.30586],
            ],
            [
                [233.545, 233.50867],
                [232.33002, 232.33957],
                [232.06425, 232.16368],
            ],
        ],
        dtype=np.float32,
    )

    result = ecmwf_ifs_hres.select_level_and_calculate_daily_mean(
        pl_ds, level=500
    )

    assert result["level"].values == 500

    np.testing.assert_allclose(result["t"].values, expected)
