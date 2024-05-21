import numpy as np
import pandas as pd
import pytest
import xarray as xr

import a6.datasets.transforms as transforms


def create_data_array(values_level_1: int, values_level_2: int) -> xr.DataArray:
    return xr.DataArray(
        [
            # day 1
            [
                # level 1
                [
                    [values_level_1, values_level_1],
                    [values_level_1, values_level_1],
                ],
                # level 2
                [
                    [values_level_2, values_level_2],
                    [values_level_2, values_level_2],
                ],
            ],
            # day 2
            [
                # level 1
                [
                    [values_level_1, values_level_1],
                    [values_level_1, values_level_1],
                ],
                # level 2
                [
                    [values_level_2, values_level_2],
                    [values_level_2, values_level_2],
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


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        (
            [1],
            np.array(
                [
                    # var1 level 1
                    [
                        [0, 0],
                        [0, 0],
                    ],
                    # var 1 level 1
                    [
                        [2, 2],
                        [2, 2],
                    ],
                ]
            ),
        ),
        (
            [1, 2],
            np.array(
                [
                    # var1 level 1
                    [
                        [0, 0],
                        [0, 0],
                    ],
                    # var 1 level 1
                    [
                        [2, 2],
                        [2, 2],
                    ],
                    # var1 level 2
                    [
                        [1, 1],
                        [1, 1],
                    ],
                    # var2 level 2
                    [
                        [3, 3],
                        [3, 3],
                    ],
                ]
            ),
        ),
    ],
)
def test_concatenate_levels_to_channels(levels, expected):
    da_1 = create_data_array(values_level_1=0, values_level_2=1)
    da_2 = create_data_array(values_level_1=2, values_level_2=3)
    ds = xr.Dataset(
        data_vars={
            "var1": da_1,
            "var2": da_2,
        },
        coords=da_1.coords,
    )

    expected = np.array(
        [
            # var1 level 1
            [
                [0, 0],
                [0, 0],
            ],
            # var 1 level 1
            [
                [2, 2],
                [2, 2],
            ],
            # var1 level 2
            [
                [1, 1],
                [1, 1],
            ],
            # var2 level 2
            [
                [3, 3],
                [3, 3],
            ],
        ]
    )

    result = transforms.xarray.concatenate_levels_to_channels(
        ds,
        time_index=0,
        levels=[1, 2],
    )

    np.testing.assert_equal(result.numpy(), expected)
