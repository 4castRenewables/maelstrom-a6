import pandas as pd
import xarray as xr

import a6.datasets.methods.normalization as normalization


def create_data_array(min_value: int, max_value: int) -> xr.DataArray:
    return xr.DataArray(
        [
            # day 1
            [
                # level 1
                [
                    [min_value, min_value],
                    [min_value, min_value],
                ],
                # level 2
                [
                    [min_value, min_value],
                    [min_value, min_value],
                ],
            ],
            # day 1
            [
                # level 1
                [
                    [min_value, min_value],
                    [min_value, min_value],
                ],
                # level 2
                [
                    [max_value, min_value],
                    [min_value, min_value],
                ],
            ],
        ],
        coords={
            "time": pd.date_range("2000-01-01", "2000-01-02", freq="1d"),
            "level": [1, 2],
            "latitude": [1.0, 0.0],
            "longitude": [0.0, 1.0],
        },
        dims=["time", "level", "latitude", "longitude"],
    )


def test_get_min_max_values_for_multiple_levels():
    da_1 = create_data_array(min_value=0, max_value=1)
    da_2 = create_data_array(min_value=2, max_value=3)
    ds = xr.Dataset(
        data_vars={
            "var1": da_1,
            "var2": da_2,
        },
        coords=da_1.coords,
    )

    expected = [
        normalization.VariableMinMax(
            name="var1",
            min=0,
            max=1,
        ),
        normalization.VariableMinMax(
            name="var2",
            min=2,
            max=3,
        ),
    ]

    result = normalization.get_min_max_values(ds)

    assert result == expected
