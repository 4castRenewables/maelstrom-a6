import datetime
import logging
import typing as t

import a6.utils as utils
import xarray as xr


logger = logging.getLogger(__name__)


def preprocess_turbine_data_and_match_with_weather_data(
    weather: xr.Dataset,
    turbine: xr.Dataset,
    power_rating: float,
    production_variable: str = "production",
    coordinates: utils.CoordinateNames = utils.CoordinateNames(),
) -> tuple[xr.Dataset, xr.Dataset]:
    """Preprocess turbine data and match time steps with weather data.

    Notes
    -----
    Preprocessing steps in order:

    1. Clean the production data by removing outliers.
    2. Get the closest grid point from the weather data.
    3. Resample the production data to hourly values.
    4. Select only the intersecting time steps from both datasets.

    """
    turbine = clean_production_data(
        turbine,
        power_rating=power_rating,
        production_variable=production_variable,
    )
    weather = get_closest_grid_point(
        weather,
        turbine=turbine,
        latitude_coordinate=coordinates.latitude,
        longitude_coordinate=coordinates.longitude,
    )
    turbine = resample_to_hourly_resolution(
        turbine,
        production_variable=production_variable,
        time_coordinate=coordinates.time,
    )
    return select_intersecting_time_steps(
        weather=weather, turbine=turbine, time_coordinate=coordinates.time
    )


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
    logger.debug(
        "Cleaning production data (variable name %s) with power rating %s",
        production_variable,
        power_rating,
    )
    data = _remove_outliers(
        data, production=production_variable, power_rating=power_rating
    )
    return data


def _remove_outliers(
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


def get_closest_grid_point(
    weather: xr.Dataset,
    turbine: xr.Dataset,
    latitude_coordinate: str = "latitude",
    longitude_coordinate: str = "longitude",
) -> xr.Dataset:
    """Get the closest grid point to the wind turbine."""
    select = {
        latitude_coordinate: turbine[latitude_coordinate],
        longitude_coordinate: turbine[longitude_coordinate],
    }
    logger.debug(
        "Getting closest grid point to wind turbine from weather data by %s",
        select,
    )
    return weather.sel(
        select,
        method="nearest",
    )


def resample_to_hourly_resolution(
    data: xr.Dataset,
    production_variable: str = "production",
    time_coordinate: str = "time",
) -> xr.Dataset:
    logger.debug("Resampling production data to hourly time resolution")
    # Resample to an hourly time series and take the mean for each hour.
    data = data.resample({time_coordinate: "1h"}).mean()
    # Remove NaNs that resulted from the resampling.
    return data.where(data[production_variable].notnull(), drop=True)


def select_intersecting_time_steps(
    weather: xr.Dataset,
    turbine: xr.Dataset,
    time_coordinate: str = "time",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Select the overlapping time steps of the datasets."""
    logger.debug("Getting intersecting time steps for weather and turbine data")
    intersection = _get_time_step_intersection(
        weather=weather,
        turbine=turbine,
        time_coordinate=time_coordinate,
    )
    select = {time_coordinate: intersection}
    logger.debug("Found intersecting time steps %s", select)
    return weather.sel(select), turbine.sel(select)


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
