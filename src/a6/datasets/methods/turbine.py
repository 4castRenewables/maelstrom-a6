import logging

import a6.datasets.coordinates as _coordinates
import a6.datasets.variables as _variables
import a6.utils as utils
import xarray as xr


logger = logging.getLogger(__name__)


@utils.log_consumption
def preprocess_turbine_data_and_match_with_weather_data(
    weather: xr.Dataset,
    turbine: xr.Dataset,
    power_rating: float,
    turbine_variables: _variables.Turbine = _variables.Turbine(),
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
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
        variables=turbine_variables,
    )
    weather = get_closest_grid_point(
        weather,
        turbine=turbine,
        latitude_coordinate=coordinates.latitude,
        longitude_coordinate=coordinates.longitude,
    )
    turbine = resample_to_hourly_resolution(
        turbine,
        variables=turbine_variables,
        time_coordinate=coordinates.time,
    )
    return select_intersecting_time_steps(
        weather=weather, turbine=turbine, time_coordinate=coordinates.time
    )


@utils.log_consumption
def clean_production_data(
    data: xr.Dataset,
    power_rating: int | float,
    variables: _variables.Turbine = _variables.Turbine(),
) -> xr.Dataset:
    """Clean the production data.

    Parameters
    ----------
    data : xr.Dataset
        Contains the power production data.
    power_rating : int or float
        Power rating of the wind turbine in kW.
        Will be used to remove outliers.
    variables : a6.datasets.variables.Turbine, optional
        Name of the power production variables.

    """
    logger.debug(
        "Cleaning production data (variables %s) with power rating %s",
        variables,
        power_rating,
    )
    data = _remove_outliers(
        data, production=variables.production, power_rating=power_rating
    )
    return data


def _remove_outliers(
    data: xr.Dataset, production: str, power_rating: int | float
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


@utils.log_consumption
def get_closest_grid_point(
    weather: xr.Dataset,
    turbine: xr.Dataset,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    """Get the closest grid point to the wind turbine."""
    select = {
        coordinates.latitude: turbine[coordinates.latitude],
        coordinates.longitude: turbine[coordinates.longitude],
    }
    logger.debug(
        "Getting closest grid point to wind turbine from weather data by %s",
        select,
    )
    return weather.sel(
        select,
        method="nearest",
    )


@utils.log_consumption
def resample_to_hourly_resolution(
    data: xr.Dataset,
    variables: _variables.Turbine = _variables.Turbine(),
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    logger.debug("Resampling production data to hourly time resolution")
    # Resample to an hourly time series and take the mean for each hour.
    data = data.resample({coordinates.time: "1h"}).mean()
    # Remove NaNs that resulted from the resampling.
    return data.where(data[variables.production].notnull(), drop=True)


@utils.log_consumption
def select_intersecting_time_steps(
    weather: xr.Dataset,
    turbine: xr.Dataset,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> tuple[xr.Dataset, xr.Dataset]:
    """Select the overlapping time steps of the datasets."""
    logger.debug("Getting intersecting time steps for weather and turbine data")
    intersection = utils.get_time_step_intersection(
        left=weather,
        right=turbine,
        time_coordinate=coordinates.time,
    )
    select = {coordinates.time: intersection}
    logger.debug("Found intersecting time steps %s", select)
    return weather.sel(select), turbine.sel(select)
