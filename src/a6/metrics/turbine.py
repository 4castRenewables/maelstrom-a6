import a6.types as types
import sklearn.metrics as metrics
import xarray as xr


def calculate_nmae(
    y_true: types.Data, y_pred: types.Data, power_rating: float
) -> xr.DataArray:
    """Calculate the NMAE of a wind turbine."""
    metrics.mean_absolute_error(y_true, y_pred) / power_rating
