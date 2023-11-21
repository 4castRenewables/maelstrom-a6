import logging
import pathlib
from collections.abc import Hashable

import sklearn.model_selection as model_selection
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.datasets.methods.turbine as _turbine
import a6.datasets.variables as _variables
import a6.features.methods as methods
import a6.training as training
import a6.types as types
import a6.utils as utils
import mlflow

logger = logging.getLogger(__name__)


@utils.log_consumption
def perform_forecast_model_grid_search(  # noqa: CFQ002
    model: type[types.Model],
    parameters: dict[str, list],
    weather: xr.Dataset,
    variables: list[Hashable],
    turbine: xr.Dataset,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
    turbine_variables: _variables.Turbine = _variables.Turbine(),
    model_variables: _variables.Model = _variables.Model(),
    log_to_mantik: bool = False,
) -> model_selection.GridSearchCV:
    """Perform grid search for a forecasting model."""
    variables = set(variables)

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
        turbine_variables=turbine_variables,
        coordinates=coordinates,
    )
    weather = (
        methods.wind.calculate_wind_speed(variables=model_variables)
        >> methods.wind.calculate_wind_direction_angle(
            variables=model_variables
        )
        >> methods.variables.drop_variables(
            names=[model_variables.u, model_variables.v]
        )
        >> methods.time.calculate_fraction_of_day(coordinates=coordinates)
        >> methods.time.calculate_fraction_of_year(coordinates=coordinates)
    ).apply_to(weather)

    cv = model_selection.LeaveOneGroupOut()
    groups = training.Groups(
        data=turbine,
        coordinates=coordinates,
        groupby="date",
    )
    scorers = training.metrics.turbine.Scorers(power_rating)

    # Remove u+v from train variables and add wind speed,
    # fraction_of_day and fraction_of_year
    train_variables = (variables - {model_variables.u, model_variables.v}) | {
        model_variables.wind_speed,
        "fraction_of_day",
        "fraction_of_year",
    }

    logger.debug(
        (
            "Performing grid search for %s with parameters %s "
            "and CV %s with groups %"
        ),
        model,
        parameters,
        cv,
        groups,
    )
    logger.info(
        (
            "Performing grid search for %s with parameters %s "
            "and input variables"
        ),
        model,
        parameters,
        train_variables,
    )
    gs = training.grid_search.perform_grid_search(
        model=model(),
        parameters=parameters,
        training_data=[weather[var] for var in train_variables],
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
