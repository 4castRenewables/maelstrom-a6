import a6.plotting.modes as modes
import pytest
import xarray as xr


@pytest.mark.parametrize("temperature", [None, "ml_ds"])
def test_plot_geopotential_height_contours(request, temperature):
    if temperature is not None:
        temperature = request.getfixturevalue(temperature)["u"].isel(time=0)

    geopotential_height = xr.DataArray(
        [
            [500, 510],
            [505, 515],
        ],
    )
    modes.plot_geopotential_height_contours(
        geopotential_height, temperature=temperature
    )
