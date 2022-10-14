import a6.features.methods as methods
import a6.plotting.modes as modes
import pytest
import xarray as xr


@pytest.fixture(scope="session")
def geopotential_height(pl_ds) -> xr.DataArray:
    return methods.calculate_geopotential_height(pl_ds["z"])


@pytest.mark.parametrize("temperature", [None, "ml_ds"])
def test_plot_geopotential_height_contours(
    request, geopotential_height, temperature
):
    if temperature is not None:
        temperature = request.getfixturevalue(temperature)

    modes.plot_geopotential_height_contours(
        geopotential_height, temperature=temperature
    )
