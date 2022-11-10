import numpy as np

import a6.utils as utils


@utils.make_functional
def reshape_spatio_temporal_numpy_array(
    data: np.ndarray,
) -> np.ndarray:
    """Reshape a `np.ndarray` that has one temporal and two spatial dimensions.

    Parameters
    ----------
    data : np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        Reshaped data with the temporal steps as rows and the spatial points
        as columns (i.e., their respective value).
        If the data has t time steps and consists of a (n, m) grid, the
        reshaped data are of shape (t x nm).

    The data must conform to the CF 1.6 conventions: The first dimension
    is the time, second is the latitudes, and third the longitudes.

    """
    shape = np.shape(data)

    time_values = shape[0]
    x_values = shape[1]
    y_values = shape[2]

    reshaped = data.reshape(time_values, x_values * y_values)
    return reshaped
