import a6.features.methods.weighting as weighting
import numpy as np
import pytest
import xarray as xr


def create_data_array(data) -> xr.DataArray:
    return xr.DataArray(
        data=[data, data],
        dims=["time", "lat", "lon"],
        coords={
            "time": ("time", [0, 1]),
            "lat": ("time", [0.0, 1.0]),
            "lon": ("time", [0.0, 1.0]),
        },
    )


@pytest.mark.parametrize(
    ("data", "latitudes", "latitudes_in_radians", "expected"),
    [
        (
            np.ones((2, 2), dtype=float),
            np.array([0, 180]),
            False,
            np.array(
                [
                    [-1, -1],
                    [1, 1],
                ],
                dtype=float,
            ),
        ),
        (
            np.ones((2, 2), dtype=float),
            np.array([0, np.pi]),
            True,
            np.array(
                [
                    [-1, -1],
                    [1, 1],
                ],
                dtype=float,
            ),
        ),
        (
            create_data_array(data=[[1, 1], [1, 1]]),
            np.array([0, 180]),
            False,
            create_data_array(
                data=[
                    [-1, -1],
                    [1, 1],
                ],
            ),
        ),
        (
            create_data_array(data=[[1, 1], [1, 1]]),
            np.array([0, np.pi]),
            True,
            create_data_array(
                data=[
                    [-1, -1],
                    [1, 1],
                ],
            ),
        ),
    ],
)
def test_weight_by_latitudes(data, latitudes, latitudes_in_radians, expected):
    result = weighting.weight_by_latitudes(
        data, latitudes=latitudes, latitudes_in_radians=latitudes_in_radians
    )

    if isinstance(data, np.ndarray):
        np.testing.assert_equal(result, expected)
    elif isinstance(data, xr.DataArray):
        xr.testing.assert_equal(result, expected)
