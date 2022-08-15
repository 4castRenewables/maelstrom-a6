import functools
import typing as t

import lifetimes.modes.methods.pca.pca as _pca
import lifetimes.utils as utils
import lifetimes.utils._types as _types
import numpy as np
import sklearn.decomposition as decomposition

PCAMethod = t.Union[decomposition.PCA, decomposition.IncrementalPCA]


@utils.log_runtime
@functools.singledispatch
def spatio_temporal_pca(
    data: t.Any,
    algorithm: t.Optional[PCAMethod] = None,
    time_coordinate: str = "time",
    latitude_coordinate: str = "latitude",
    x_coordinate: t.Optional[str] = None,
    y_coordinate: t.Optional[str] = None,
) -> _pca.PCA:
    """Perform a spatio-temporal PCA.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Spatial timeseries data.
        Must be passed as an positional argument, i.e. as
        ```python
        spatio_temporal_pca(data, ...)
        ```
        Passing as a keyword argument (i.e. `data=data`) will raise a
        `TypeError`.
    algorithm : sklearn.decomposition.PCA or IncrementalPCA, default=PCA
        Method to use for the PCA.
    time_coordinate : str, default="time"
        Name of the time coordinate.
        This is required to reshape the data for the PCA.
    latitude_coordinate : str, default="latitude"
        Name of the latitudinal coordinate.
        This is required for weighting the data by latitude before the PCA.
    x_coordinate : str, optional
        Name of the x-coordinate of the grid.
        If `None`, CF 1.6 convention will be assumed, i.e. `"longitude"`.
    y_coordinate : str, optional
        Name of the y-coordinate of the grid.
        If `None`, CF 1.6 convention will be assumed, i.e. `"latitude"`.

    Returns
    -------
    PCA
        Contains the PCs, eigenvalues, variance ratios etc.
        Also holds an instance of the original data.

    Performs a Singular Spectrum Analysis. To do so, the data have to be
    reshaped into a matrix consisting of the locations and their respective
    value as columns, and the time steps as rows. I.e., if the data consist of
    a measured quantity on a (m x n) grid with p time steps, the resulting
    matrix is of size (p x mn).

    For reference see e.g. Jolliffe I. T., Principal Component Analysis, 2ed.,
    Springer, 2002, page 302 ff.

    """
    dimensions, data, pca = _apply_pca(
        data=data,
        algorithm=algorithm,
        time_coordinate=time_coordinate,
        latitude_coordinate=latitude_coordinate,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )
    return _pca.PCA(
        pca=pca,
        reshaped=data,
        dimensions=dimensions,
    )


def _apply_pca(
    data: _types.Data,
    algorithm: t.Optional[PCAMethod] = None,
    time_coordinate: str = "time",
    latitude_coordinate: str = "latitude",
    x_coordinate: t.Optional[str] = None,
    y_coordinate: t.Optional[str] = None,
) -> tuple[utils.Dimensions, np.ndarray, PCAMethod]:
    if algorithm is None:
        algorithm = decomposition.PCA()

    (dimensions, data) = _reshape_data(
        data=data,
        time_coordinate=time_coordinate,
        latitude_coordinate=latitude_coordinate,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )
    pca: PCAMethod = algorithm.fit(data)
    return dimensions, data, pca


def _reshape_data(
    data: _types.Data,
    time_coordinate: str,
    latitude_coordinate: str,
    x_coordinate: t.Optional[str],
    y_coordinate: t.Optional[str],
) -> tuple[utils.Dimensions, np.ndarray]:
    dimensions = utils.Dimensions.from_xarray(
        data, time_dimension=time_coordinate
    )
    data = utils.weight_by_latitudes(
        data=data,
        latitudes=latitude_coordinate,
        use_sqrt=True,
    )
    data = utils.reshape_spatio_temporal_xarray_data(
        data=data,
        time_coordinate=None,  # Set to None to avoid memory excess in function
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )
    return (
        dimensions,
        data,
    )
