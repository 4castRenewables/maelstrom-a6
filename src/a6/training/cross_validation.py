import a6.training.metrics as metrics
import a6.types as types
import a6.utils as utils
import numpy as np
import sklearn.model_selection as model_selection

import mlflow


def cross_validation(
    model: object,
    X: types.XarrayData,
    y: types.XarrayData,
    power_rating: float,
    log_to_mantik: bool = True,
) -> dict[str, np.ndarray]:
    """Perform cross validation."""
    cv = model_selection.cross_validate(
        model,
        X=utils.transpose(X),
        y=utils.transpose(y),
        scoring=metrics.turbine.make_scorers(power_rating),
        n_jobs=utils.get_cpu_count(),
    )

    if log_to_mantik:
        _log_result(model=model, cv=cv)

    return cv


def _log_result(model: object, cv: dict[str, np.ndarray]) -> None:
    mae = cv["test_mae"]
    nmae = cv["test_nmae"]
    rmse = cv["test_rmse"]
    nrmse = cv["test_nrmse"]
    mlflow.log_param("model", model.__class__.__name__)
    mlflow.log_metric("fit_time_mean", cv["fit_time"].mean())
    mlflow.log_metric("fit_time_std", cv["fit_time"].std())

    for metric, value in [
        ("mae", mae),
        ("nmae", nmae),
        ("rmse", rmse),
        ("nrmse", nrmse),
    ]:
        mlflow.log_metric(f"{metric}_mean", value.mean())
        mlflow.log_metric(f"{metric}_median", np.median(value))
        mlflow.log_metric(f"{metric}_std", value.std())
