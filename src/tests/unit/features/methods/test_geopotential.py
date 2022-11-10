import xarray as xr

import a6.features.methods.geopotential as geopotential


def test_calculate_geopotential_height(pl_ds):
    result = geopotential.calculate_geopotential_height(
        pl_ds, non_functional=True
    )

    assert isinstance(result, xr.Dataset)
    assert "z_h" in result.data_vars
