import datetime

import numpy as np
import pytest
import xarray as xr

import a6.datasets.methods.mask as mask


def create_data_array(levels: int, time_steps: int) -> xr.DataArray:
    if levels == 0 or time_steps == 0:
        raise RuntimeError(
            "Cannot create dataset with zero levels or time steps"
        )

    base_data = [
        [np.nan, 1],
        [1, 1],
    ]

    if time_steps == 1:
        data = base_data
        coords = {"time": datetime.datetime(2000, 1, 1)}
        dims = []
    else:
        data = [base_data for _ in range(time_steps)]
        coords = {
            "time": [
                datetime.datetime(2000, 1, i + 1) for i in range(time_steps)
            ]
        }
        dims = ["time"]

    if levels == 1:
        coords = coords | {
            "level": 1,
        }
    else:
        data = [data for _ in range(levels)]
        coords = coords | {
            "level": [i for i in range(levels)],
        }
        dims.append("level")

    coords = coords | {
        "latitude": [1.0, 0.0],
        "longitude": [0.0, 1.0],
    }
    dims.extend(["latitude", "longitude"])

    return xr.DataArray(
        data,
        coords=coords,
        dims=dims,
    )


@pytest.mark.parametrize(
    ("levels", "time_steps"), [(1, 1), (2, 1), (1, 2), (2, 2)]
)
def test_get_min_max_values_for_multiple_levels(levels, time_steps):
    da_1 = create_data_array(levels=levels, time_steps=time_steps)
    da_2 = create_data_array(levels=levels, time_steps=time_steps)

    ds = xr.Dataset(
        data_vars={
            "var1": da_1,
            "var2": da_2,
        },
        coords=da_1.coords,
    )

    result = mask.set_nans_to_mean(ds)

    assert not any(np.isnan(result[var]).any() for var in result.data_vars)
