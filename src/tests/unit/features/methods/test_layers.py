import pytest
import xarray as xr

import a6.features.methods.layers as layers


@pytest.mark.parametrize(
    ("n_quantities", "expected"), [(1, 1), (2, 2), (3, 3), (4, 4)]
)
def test_get_number_of_input_channels_for_dataset(da, n_quantities, expected):
    ds = xr.Dataset(
        data_vars={f"var{i}": da for i in range(n_quantities)},
        coords=da.coords,
    )
    result = layers.get_number_of_input_channels(ds)

    assert result == expected


def test_get_number_of_input_channels_for_data_array(da):
    expected = 1

    result = layers.get_number_of_input_channels(da)

    assert result == expected
