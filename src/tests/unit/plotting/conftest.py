import pytest
import xarray as xr


@pytest.fixture
def time_series_data() -> xr.DataArray:
    return xr.DataArray(
        [
            1,
            1,
            2,
            6,
            8,
            5,
            5,
            7,
            8,
            8,
            1,
            1,
            4,
            5,
            5,
            0,
            0,
            0,
            1,
            1,
            4,
            4,
            5,
            1,
            3,
            3,
            4,
            5,
            4,
            1,
            1,
        ]
    )
