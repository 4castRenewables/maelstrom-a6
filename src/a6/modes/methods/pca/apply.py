import a6.features.methods as methods
import a6.modes.methods.pca.pca as _pca
import a6.types as types
import a6.utils as utils
import sklearn.decomposition as decomposition
import xarray as xr


@utils.log_consumption
def spatio_temporal_pca(
    data: types.DataND,
    algorithm: _pca.PCAMethod | None = None,
    coordinates: utils.CoordinateNames = utils.CoordinateNames(),
    x_coordinate: str | None = None,
    y_coordinate: str | None = None,
) -> _pca.PCA:
    """Perform a spatio-temporal PCA.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Spatial timeseries data.
    algorithm : sklearn.decomposition.PCA or IncrementalPCA, default=PCA
        Method to use for the PCA.
    coordinates : a6.utils.CoordinateNames
        Names of the coordinates.
        These are required
            - to reshape the data for the PCA.
            - for weighting the data by latitude before the PCA.
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
    dimensions, data, sklearn_pca = _apply_pca(
        data=data,
        algorithm=algorithm,
        coordinates=coordinates,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )
    return _pca.PCA(
        sklearn_pca=sklearn_pca,
        reshaped=data,
        dimensions=dimensions,
    )


def _apply_pca(
    data: types.DataND,
    coordinates: utils.CoordinateNames,
    algorithm: _pca.PCAMethod | None = None,
    x_coordinate: str | None = None,
    y_coordinate: str | None = None,
) -> tuple[utils.SpatioTemporalDimensions, xr.DataArray, _pca.PCAMethod]:
    if algorithm is None:
        algorithm = decomposition.PCA()

    (dimensions, data) = _reshape_and_standardize_data(
        data=data,
        coordinates=coordinates,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )
    sklearn_pca: _pca.PCAMethod = algorithm.fit(data)
    return dimensions, data, sklearn_pca


def _reshape_and_standardize_data(
    data: types.DataND,
    coordinates: utils.CoordinateNames,
    x_coordinate: str | None,
    y_coordinate: str | None,
) -> tuple[utils.SpatioTemporalDimensions, xr.DataArray]:
    dimensions = utils.SpatioTemporalDimensions.from_xarray(
        data,
        coordinates=coordinates,
    )
    data = methods.weighting.weight_by_latitudes(
        data=data,
        latitudes=coordinates.latitude,
        use_sqrt=True,
    )
    data = methods.reshape.xarray.reshape_spatio_temporal_data(
        data=data,
        time_coordinate=None,  # Set to None to avoid memory excess in function
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )
    data = methods.standardization.standardize_features(data)
    return (
        dimensions,
        data,
    )
