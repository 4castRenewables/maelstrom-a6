import datetime

import numpy as np
import pandas as pd
import xarray as xr

import lifetimes.modes.methods.appearances as appearances


def test_determine_appearances_of_modes():
    # Assume a time series with 6 time steps and 2 different modes that each
    # lasts for 3 consecutive days (time units).
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2000, 1, 6)
    modes = [0, 0, 0, 1, 1, 1]
    dates = pd.date_range(start=start, end=end, freq="1d")
    time_series = xr.DataArray(data=modes, coords={"time": dates})

    expected = [
        appearances.Mode(
            label=0,
            appearances=[
                appearances.Appearance(
                    start=np.datetime64(datetime.datetime(2000, 1, 1)),
                    end=np.datetime64(datetime.datetime(2000, 1, 3)),
                    index=appearances.AppearanceIndex(start=0, end=2),
                ),
            ],
            statistics=appearances.Statistics(
                total=1,
                duration_mean=3.0,
                duration_std=0.0,
            ),
        ),
        appearances.Mode(
            label=1,
            appearances=[
                appearances.Appearance(
                    start=np.datetime64(datetime.datetime(2000, 1, 4)),
                    end=np.datetime64(datetime.datetime(2000, 1, 6)),
                    index=appearances.AppearanceIndex(start=3, end=5),
                ),
            ],
            statistics=appearances.Statistics(
                total=1,
                duration_mean=3.0,
                duration_std=0.0,
            ),
        ),
    ]

    result = appearances.determine_lifetimes_of_modes(
        time_series,
        time_coordinate="time",
    )

    assert result == expected
