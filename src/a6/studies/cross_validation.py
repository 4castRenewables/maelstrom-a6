import functools
import typing as t

import a6.metrics.turbine as turbine
import a6.types as types
import a6.utils as utils
import numpy as np
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection

import mlflow


def cross_validation_on_turbine_data(
    model: object,
    X: types.Data,
    y: types.Data,
    power_rating: float,
    log_to_mantik: bool = True,
) -> dict[str, np.ndarray]:
    """Perform cross validation."""
    cv = model_selection.cross_validate(
        model,
        X=X,
        y=y,
        scoring=_make_scorings(power_rating=power_rating),
        n_jobs=utils.get_cpu_count(),
    )

    if log_to_mantik:
        _log_result(model=model, cv=cv)

    return cv


def _make_scorings(power_rating: float) -> dict[str, t.Callable]:
    return {
        "mae": metrics.make_scorer(metrics.mean_absolute_error),
        "nmae": metrics.make_scorer(
            functools.partial(turbine.calculate_nmae, power_rating=power_rating)
        ),
    }


def _log_result(model: object, cv: dict[str, np.ndarray]) -> None:
    mae = cv["test_mae"]
    nmae = cv["test_nmae"]
    mlflow.log_param("model", model.__class__.__name__)
    mlflow.log_metric("fit_time_mean", cv["fit_time"].mean())
    mlflow.log_metric("fit_time_std", cv["fit_time"].std())

    for metric, value in [("mae", mae), ("nmae", nmae)]:
        mlflow.log_metric(f"{metric}_mean", value.mean())
        mlflow.log_metric(f"{metric}_median", np.median(value))
        mlflow.log_metric(f"{metric}_std", value.std())
