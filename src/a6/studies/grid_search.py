import logging
import pathlib

import a6.datasets.coordinates as _coordinates
import a6.datasets.methods.turbine as _turbine
import a6.datasets.variables as _variables
import a6.features.methods.wind as wind
import a6.training as training
import a6.types as types
import a6.utils as utils
import sklearn.model_selection as model_selection
import xarray as xr

import mlflow

logger = logging.getLogger(__name__)


@utils.log_consumption
def perform_forecast_model_grid_search(  # noqa: CFQ002
    model: type[types.Model],
    parameters: dict[str, list],
    weather: xr.Dataset,
    turbine: xr.Dataset,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
    turbine_variables: _variables.Turbine = _variables.Turbine(),
    model_variables: _variables.Model = _variables.Model(),
    log_to_mantik: bool = False,
) -> model_selection.GridSearchCV:
    """Perform grid search for a forecasting model."""
    if log_to_mantik:
        mlflow.sklearn.autolog(log_models=False)

    power_rating = turbine_variables.read_power_rating(turbine)
    (
        weather,
        turbine,
    ) = _turbine.preprocess_turbine_data_and_match_with_weather_data(
        weather=weather,
        turbine=turbine,
        power_rating=power_rating,
        production_variable=turbine_variables.production,
        coordinates=coordinates,
    )
    wind_speed = wind.calculate_wind_speed(weather, variables=model_variables)

    cv = model_selection.LeaveOneGroupOut()
    groups = training.Groups(
        data=turbine,
        time_coordinate=coordinates.time,
        groupby="date",
    )
    scorers = training.metrics.turbine.Scorers(power_rating)

    logger.debug(
        "Performing grid search for %s with parameters %s "
        "and cross validation %s with groups %s",
        model,
        parameters,
        cv,
        groups,
    )
    gs = training.grid_search.perform_grid_search(
        model=model(),
        parameters=parameters,
        training_data=wind_speed,
        target_data=turbine[turbine_variables.production],
        cv=cv,
        groups=groups.labels,
        scorers=scorers.to_dict(),
        refit=scorers.main_score,
    )

    scores = groups.evaluate_cross_validation(
        gs.cv_results_, scores=scorers.scores
    )
    scores_with_coords = scores.assign_coords(
        {
            coordinates.latitude: turbine[coordinates.longitude],
            coordinates.longitude: turbine[coordinates.latitude],
        }
    )
    if log_to_mantik:
        _log_scores_as_netcdf(
            scores_with_coords,
            name=f"{turbine_variables.get_turbine_name(turbine)}-scores.nc",
        )

    return gs


def _log_scores_as_netcdf(scores: xr.Dataset, name: str) -> None:
    path = pathlib.Path(name)
    scores.to_netcdf(path)
    mlflow.log_artifact(path.as_posix())
    path.unlink()
