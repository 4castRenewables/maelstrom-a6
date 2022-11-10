import datetime

import pandas as pd
import xarray as xr

import a6.modes.methods.appearances as appearances


def test_determine_lifetimes_of_modes(mode_appearances):
    # Assume a time series with 6 time steps and 2 different modes that each
    # lasts for 3 consecutive days (time units).
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2000, 1, 6)
    modes = [0, 0, 0, 1, 1, 1]
    dates = pd.date_range(start=start, end=end, freq="1d")
    time_series = xr.DataArray(data=modes, coords={"time": dates})

    result = appearances.determine_lifetimes_of_modes(
        time_series,
    )

    assert result.modes == mode_appearances.modes
