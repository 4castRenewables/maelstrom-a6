import datetime
import typing as t

import xarray as xr


def clean_production_data(
    data: xr.Dataset,
    power_rating: t.Union[int, float],
    production_variable: str = "production",
) -> xr.Dataset:
    """Clean the production data.

    Parameters
    ----------
    data : xr.Dataset
        Contains the power production data.
    power_rating : int or float
        Power rating of the wind turbine in kW.
        Will be used to remove outliers.
    production_variable : str, default="production"
        Name of the power production variable.


    """
    data = _remote_outliers(
        data, production=production_variable, power_rating=power_rating
    )
    return data


def _remote_outliers(
    data: xr.Dataset, production: str, power_rating: t.Union[int, float]
) -> xr.Dataset:
    return data.where(
        (
            # Find indexes where |P| < power_rating
            (abs(data[production]) < 1.1 * power_rating)
            &
            # and such where P > 0
            (data[production] > 0)
            &
            # and such where P is not NaN.
            data[production].notnull()
        ),
        drop=True,
    )


def resample_to_hourly_resolution(
    data: xr.Dataset,
    production_variable: str = "time",
    time_coordinate: str = "time",
) -> xr.Dataset:
    # Resample to an hourly time series and take the mean for each hour.
    data = data.resample({time_coordinate: "1h"}).mean()
    # Remove NaNs that resulted from the resampling.
    return data.where(data[production_variable].notnull(), drop=True)


def get_closest_grid_point(
    weather: xr.Dataset,
    turbine: xr.Dataset,
    latitudinal_coordinate: str = "latitude",
    longitudinal_coordinate: str = "longitude",
) -> xr.Dataset:
    """Get the closest grid point to the wind turbine."""
    return weather.sel(
        {
            latitudinal_coordinate: turbine[latitudinal_coordinate],
            longitudinal_coordinate: turbine[longitudinal_coordinate],
        },
        method="nearest",
    )


def select_overlapping_time_steps(
    weather: xr.Dataset,
    turbine: xr.Dataset,
    time_coordinate: str = "time",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Select the overlapping time steps of the datasets."""
    intersection = _get_time_step_intersection(
        weather=weather,
        turbine=turbine,
        time_coordinate=time_coordinate,
    )
    select = {time_coordinate: intersection}
    return (weather.sel(select), turbine.sel(select))


def _get_time_step_intersection(
    weather: xr.Dataset,
    turbine: xr.Dataset,
    time_coordinate: str,
) -> list[datetime.datetime]:
    # Create sets of the time steps to allow set theory operations.
    weather_time_stamps = set(weather[time_coordinate].values)
    turbine_time_stamps = set(turbine[time_coordinate].values)
    intersection = weather_time_stamps & turbine_time_stamps
    return sorted(intersection)
