import numpy as np
import pytest
import xarray as xr

import a6.testing as testing
import a6.testing._ellipse as _ellipse


def create_data_array_with_data(
    grid: testing.TestGrid, data: np.ndarray
) -> xr.DataArray:
    da = xr.DataArray(
        data=data,
        dims=grid.coordinates,
        coords=grid.xarray_coords_dict,
    )
    return da


@pytest.mark.parametrize(
    ("grid", "a", "b", "center", "rotate", "expected"),
    [
        # Test case: circle of radius 1 in the center of a 3x3 grid
        (
            testing.TestGrid(rows=3, columns=3),
            2 / 3,
            2 / 3,
            (0.0, 0.0),
            False,
            np.array(
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                ],
                dtype=float,
            ),
        ),
        # Test case: circle of radius 1 in the center of a 3x5 grid
        (
            testing.TestGrid(rows=3, columns=5),
            2 / 3,
            2 / 3,
            (0.0, 0.0),
            False,
            np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                ],
                dtype=float,
            ),
        ),
        # Test case: ellipse with (a, b) = (2, 1) in the center of 5x5 grid
        (
            testing.TestGrid(rows=5, columns=5),
            2 / 2.5,
            1 / 2.5,
            (0.0, 0.0),
            False,
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=float,
            ),
        ),
        # Test case: ellipse with (a, b) = (2, 1) in the center of 5x5 grid,
        # rotated by 90 degrees
        (
            testing.TestGrid(rows=5, columns=5),
            2 / 2.5,
            1 / 2.5,
            (0.0, 0.0),
            True,
            np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                ],
                dtype=float,
            ),
        ),
        # Test case: circle centered at (1, 1) on a 5x5 grid
        (
            testing.TestGrid(rows=5, columns=5),
            1 / 2.5,
            1 / 2.5,
            (1 / 2.5, 1 / 2.5),
            False,
            np.array(
                [
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=float,
            ),
        ),
    ],
)
def test_create_ellipse_on_grid(grid, a, b, center, rotate, expected):
    expected_data_array = create_data_array_with_data(grid=grid, data=expected)
    result = _ellipse.create_ellipse_on_grid(
        grid=grid,
        a=a,
        b=b,
        center=center,
        rotate=rotate,
    )

    xr.testing.assert_equal(result, expected_data_array)
