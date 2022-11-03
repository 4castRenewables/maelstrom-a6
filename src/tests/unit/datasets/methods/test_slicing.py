import a6.datasets.methods.slicing as slicing
import pytest
import xarray as xr


@pytest.mark.parametrize(
    ("slice_from", "slice_until"),
    [
        (None, 12),
        (0, 12),
        (2, 10),
    ],
)
def test_slice_dataset(slice_from, slice_until):
    data = [1 for _ in range(0, 2 * slice_until)]

    start = 0 if slice_from is None else slice_from
    expected_data = [1 for _ in range(start, slice_until)]

    def _create_dataset(d):
        return xr.Dataset(
            data_vars={
                "variable": xr.DataArray(d),
            },
        )

    dataset = _create_dataset(data)
    expected = _create_dataset(expected_data)

    result = slicing.slice_dataset(
        dataset,
        dimension="dim_0",
        slice_from=slice_from,
        slice_until=slice_until,
        non_functional=True,
    )

    xr.testing.assert_equal(result, expected)
