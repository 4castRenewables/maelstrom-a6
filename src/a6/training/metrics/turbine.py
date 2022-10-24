import a6.types as types
import numpy as np
import sklearn.metrics as metrics


class Scorers:
    """MAE, NMAE, RMSE and NRMSE scorers."""

    def __init__(self, power_rating: float):
        self._mae = metrics.make_scorer(
            metrics.mean_absolute_error, greater_is_better=False
        )
        self._nmae = metrics.make_scorer(
            calculate_nmae, greater_is_better=False, power_rating=power_rating
        )
        self._rmse = metrics.make_scorer(
            calculate_rmse, greater_is_better=False
        )
        self._nrmse = metrics.make_scorer(
            calculate_nrmse, greater_is_better=False, power_rating=power_rating
        )

    @property
    def main_score(self) -> str:
        """Return the main score function for the best estimator."""
        return "nrmse"

    @property
    def scores(self) -> list[str]:
        """Return the score names."""
        return list(self.to_dict())

    def to_dict(self) -> types.Scorers:
        """Return as dict as expected by sklearn API."""
        return {
            "mae": self._mae,
            "nmae": self._nmae,
            "rmse": self._rmse,
            "nrmse": self._nrmse,
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
