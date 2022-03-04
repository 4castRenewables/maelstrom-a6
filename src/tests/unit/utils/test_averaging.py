import datetime

import lifetimes.utils.averaging as averaging
import numpy as np
import pandas as pd
import pytest
import xarray as xr

START = datetime.datetime(2000, 1, 1)
END = datetime.datetime(2000, 2, 1)
START_DATA = [
    [1, 2],
    [1, 2],
]
END_DATA = [
    [3, 4],
    [3, 4],
]
LAT = [0.0, 1.0]
LON = [0.0, 1.0]


@pytest.mark.parametrize(
    "time_coordinate",
    # Test function with different time coordinate names.
    ["time", "time2"],
)
def test_calculate_daily_averages(time_coordinate):
    # Rename time coordinate.
    dataset = create_ingoing_dataset(time_coordinate)
    expected = create_expected_dataset(time_coordinate)

    expected_dates = expected.coords[time_coordinate].values

    result = averaging.calculate_daily_averages(
        dataset, time_coordinate=time_coordinate
    )
    result_dates = result.coords[time_coordinate].values

    xr.testing.assert_equal(result, expected)
    # Resulting dates must be `datetime64[ms]`. The type of the dates is not
    # checked for equality by `xr.testing.assert_equal`.
    np.testing.assert_equal(result_dates, expected_dates)


def create_ingoing_dataset(time_coordinate: str) -> xr.Dataset:
    """Create a test dataset.

    Will consist of 2 months with hourly data for one day each. The data are
    defined on a 2x2 grid.

    """
    dates_1 = pd.date_range(start=START, end=START, freq="1h")
    dates_2 = pd.date_range(start=END, end=END, freq="1h")
    dates = dates_1.tolist() + dates_2.tolist()
    data_1 = [START_DATA for _ in range(dates_1.size)]
    data_2 = [END_DATA for _ in range(dates_1.size)]
    data = data_1 + data_2
    return _create_dataset(
        data=data,
        time_coordinate=time_coordinate,
        dates=dates,
    )


def create_expected_dataset(time_coordinate: str) -> xr.Dataset:
    """Create a test dataset.

    Will consist of 2 months with hourly data for one day each. The data are
    defined on a 2x2 grid.

    """
    dates = np.array([START, END], dtype="datetime64[ms]")
    data = [START_DATA, END_DATA]
    return _create_dataset(
        data=data,
        time_coordinate=time_coordinate,
        dates=dates,
    )


def _create_dataset(
    data: list, time_coordinate: str, dates: list
) -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "t": ([time_coordinate, "lat", "lon"], np.array(data, dtype=float)),
        },
        coords={
            time_coordinate: ([time_coordinate], dates),
            "lat": (["lat"], LAT),
            "lon": (["lon"], LON),
        },
    )
