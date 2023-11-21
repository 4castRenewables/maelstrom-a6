import logging

import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.datasets.variables as _variables
import a6.utils as utils


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
    turbine = (
        clean_production_data(
            power_rating=power_rating,
            variables=turbine_variables,
        )
        >> resample_to_hourly_resolution(
            variables=turbine_variables,
            coordinates=coordinates,
        )
    ).apply_to(turbine)
    weather = get_closest_grid_point(
        turbine=turbine,
        coordinates=coordinates,
    ).apply_to(weather)
    return select_intersecting_time_steps(
        weather=weather,
        turbine=turbine,
        coordinates=coordinates,
        return_turbine=True,
        non_functional=True,
    )


@utils.log_consumption
@utils.make_functional
def clean_production_data(
    data: xr.Dataset,
    power_rating: int | float,
    variables: _variables.Turbine = _variables.Turbine(),
) -> xr.Dataset:
    """Clean the production data by removing outliers.

    Parameters
    ----------
    data : xr.Dataset
        Contains the power production data.
    power_rating : int or float
        Power rating of the wind turbine in kW.
        Will be used to remove outliers.
    variables : a6.datasets.variables.Turbine, optional
        Name of the power production variables.

    Notes
    -----
    Outliers are data points matching one of the below criterions:

    1. Power production is higher than the power rating.
    2. Production is below 0.
    3. Production is NaN.

    """
    logger.debug(
        "Cleaning production data (variables %s) with power rating %s",
        variables,
        power_rating,
    )
    return _remove_outliers(
        production=variables.production, power_rating=power_rating
    ).apply_to(data)


@utils.make_functional
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
@utils.make_functional
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
@utils.make_functional
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
@utils.make_functional
def select_intersecting_time_steps(
    weather: xr.Dataset,
    turbine: xr.Dataset,
    return_turbine: bool = False,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset | tuple[xr.Dataset, xr.Dataset]:
    """Select the overlapping time steps of the datasets."""
    logger.debug("Getting intersecting time steps for weather and turbine data")
    intersection = utils.get_time_step_intersection(
        left=weather,
        right=turbine,
        coordinates=coordinates,
    )
    select = {coordinates.time: intersection}
    logger.debug("Found intersecting time steps %s", select)
    if return_turbine:
        return weather.sel(select), turbine.sel(select)
    return weather.sel(select)
