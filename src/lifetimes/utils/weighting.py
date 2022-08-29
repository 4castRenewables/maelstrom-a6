import typing as t

import lifetimes.utils._types as _types
import lifetimes.utils.logging as logging
import numpy as np

Latitudes = t.Union[str, np.ndarray]


@logging.log_consumption
def weight_by_latitudes(
    data: _types.Data,
    latitudes: Latitudes,
    latitudes_in_radians: bool = False,
    use_sqrt: bool = False,
) -> _types.Data:
    """Weight grid data with cosine of the latitude.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Input data.
    latitudes : np.ndarray or str
        Ordered latitude values or name of the latitudinal coordinate.
        If `data` is a `np.ndarray`, `np.ndarray` containing the latitudes
        in degrees. If `data` is an `xr.DataArray`, name of the latitude
        coordinate.
    latitudes_in_radians : bool, default=False
        Whether the latitudes are given in radians.
    use_sqrt : bool, default=False
        Whether to use the square root of `cos(phi`).

    Returns
    -------
    np.ndarray or xr.DataArray
        Each value weighted by its latitude.

    The data must be defined on a regular grid in a specific orientation. I.e.,
    the x-direction of the grid must be in longitudinal direction (East), and
    the y-direction in latitudinal direction (North).
    The latitudinal weights are multiplied as a column vector column-wise onto
    the grid. Hence, if `latitudes` is a `np.ndarray`, it must be ordered
    (either descending or ascending).

    """
    if isinstance(data, np.ndarray) and isinstance(latitudes, str):
        raise ValueError(
            "If input data is of type np.ndarray, 'latitudes' must be as well"
        )
    if isinstance(latitudes, str):
        latitudes = data[latitudes].values
    weights = _calculate_latitudinal_weights(
        latitudes, is_radians=latitudes_in_radians
    )
    if use_sqrt:
        return data * np.sqrt(weights)
    return data * weights


def _calculate_latitudinal_weights(
    latitudes: np.ndarray, is_radians: bool
) -> np.ndarray:
    latitudes_descending = _order_latitudes_descending(latitudes)
    if not is_radians:
        latitudes_in_radians = np.radians(latitudes_descending)
        weights = np.cos(latitudes_in_radians)
    else:
        weights = np.cos(latitudes_descending)
    return _reshape_to_column_vector(weights)


def _order_latitudes_descending(latitudes: np.ndarray) -> np.ndarray:
    if _latitudes_in_ascending_order(latitudes):
        return latitudes[::-1]
    return latitudes


def _latitudes_in_ascending_order(latitudes: np.ndarray) -> bool:
    return latitudes[0] < latitudes[-1]


def _reshape_to_column_vector(vector: np.ndarray) -> np.ndarray:
    return vector[:, None]
