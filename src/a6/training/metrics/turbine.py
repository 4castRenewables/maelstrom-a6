import functools

import a6.types as types
import numpy as np
import sklearn.metrics as metrics


def make_scorers(power_rating: float) -> types.Scorers:
    """Make MAE, NMAE, RMSE and NRMSE scorers."""
    return {
        "mae": metrics.make_scorer(metrics.mean_absolute_error),
        "nmae": metrics.make_scorer(
            functools.partial(calculate_nmae, power_rating=power_rating)
        ),
        "rmse": metrics.make_scorer(calculate_rmse),
        "nrmse": metrics.make_scorer(
            functools.partial(calculate_nrmse, power_rating=power_rating)
        ),
    }


def calculate_nmae(
    y_true: types.Data, y_pred: types.Data, power_rating: float
) -> float:
    """Calculate the normalized mean-average error of a wind turbine."""
    return (
        metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred) / power_rating
    )


def calculate_nrmse(
    y_true: types.Data, y_pred: types.Data, power_rating: float
) -> float:
    """Calculate the normalized root-mean-square error of a wind turbine."""
    return calculate_rmse(y_true=y_true, y_pred=y_pred) / power_rating


def calculate_rmse(y_true: types.Data, y_pred: types.Data) -> float:
    """Calculate the root-mean-square error of a wind turbine."""
    return np.sqrt(metrics.mean_squared_error(y_true=y_true, y_pred=y_pred))
