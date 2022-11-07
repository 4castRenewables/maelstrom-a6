import a6.utils.functional as functional
import xarray as xr


@functional.make_functional
def add(data: xr.DataArray, number: int) -> xr.DataArray:
    return data + number


@functional.make_functional
def multiply(data: xr.DataArray, number: int) -> xr.DataArray:
    return data * number


@functional.make_functional
def subtract(data: xr.DataArray, number: int) -> xr.DataArray:
    return data - number


def test_make_functional():
    da = xr.DataArray(
        [
            [1, 2],
            [3, 4],
        ],
    )

    pipe = add(number=1) >> multiply(number=2) >> subtract(number=3)

    expected = add(da, number=1, non_functional=True)
    expected = multiply(expected, number=2, non_functional=True)
    expected = subtract(expected, number=3, non_functional=True)

    result = pipe.apply_to(da)

    xr.testing.assert_equal(result, expected)
