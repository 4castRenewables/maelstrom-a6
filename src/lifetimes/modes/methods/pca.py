import typing as t

import lifetimes.utils
import numpy as np
import sklearn.decomposition as decomposition
import xarray as xr

PCAMethod = t.Union[
    t.Type[decomposition.PCA], t.Type[decomposition.IncrementalPCA]
]


class PCA:
    """Wrapper for `sklearn.decomposition.PCA`."""

    def __init__(
        self,
        pca: PCAMethod,
        reshaped: np.ndarray,
        original_shape: tuple,
        time_series: xr.DataArray,
    ):
        """Wrap `sklearn.decomposition.PCA`.

        Parameters
        ----------
        pca : sklearn.decomposition.PCA
        reshaped : np.ndarray
            The reshaped, original data used for PCA.
        original_shape : tuple
            Original shape of the data the PCA was performed on.
        time_series : xr.DataArray
            The timeseries of the original data.

        """
        self._pca = pca
        self._reshaped = reshaped
        self._original_shape = original_shape
        self._time_series = time_series

    @property
    def timeseries(self) -> xr.DataArray:
        return self._time_series

    @property
    def components(self) -> np.ndarray:
        """Return the principal components (EOFs)."""
        return self._pca.components_

    @property
    def n_components(self) -> int:
        """Return the number of principal components (EOFs)."""
        return self._pca.components_.shape[0]

    @property
    def components_in_original_shape(self) -> np.ndarray:
        """Return the principal components (EOFs)."""
        return self._to_original_shape(self.components)

    @property
    def components_varimax_rotated(self) -> np.ndarray:
        """Return the principal components (EOFs)."""
        return _perform_varimax_rotation(self.components)

    @property
    def components_varimax_rotated_in_original_shape(self) -> np.ndarray:
        """Return the principal components (EOFs)."""
        return self._to_original_shape(self.components_varimax_rotated)

    def _to_original_shape(self, data: np.ndarray) -> np.ndarray:
        return data.reshape(self._original_shape)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Return the corresponding eigenvalues."""
        return self._pca.explained_variance_

    @property
    def loadings(self) -> np.ndarray:
        """Return the loadings of each PC."""
        return (self._pca.components_.T * np.sqrt(self.eigenvalues)).T

    @property
    def variance_ratios(self) -> np.ndarray:
        """Return the explained variance ratios."""
        return self._pca.explained_variance_ratio_

    @property
    def cumulative_variance_ratios(self) -> np.ndarray:
        """Return the cumulative variance ratios."""
        return np.cumsum(self.variance_ratios)

    @lifetimes.utils.log_runtime
    def transform(self, n_components: t.Optional[int] = None) -> np.ndarray:
        """Transform the given data into the vector space of the PCs."""
        return self._transform(self.components, n_components=n_components)

    @lifetimes.utils.log_runtime
    def transform_with_varimax_rotation(
        self, n_components: t.Optional[int] = None
    ) -> np.ndarray:
        """Transform the given data into the
        vector space of the varimax-rotated PCs."""
        # TODO: Do research on whether the varimax rotation has to be performed
        # on the reduced number of PCs to achieve `n_components` or on all PCs
        # to then select `n_copoments` of the varimax-rotated PCs. Currently,
        # the former is implemented. Thus, varimax rotation is very efficient.
        return self._transform(
            self.components_varimax_rotated, n_components=n_components
        )

    def _transform(self, data: np.ndarray, n_components: int) -> np.ndarray:
        return _transform_data_into_vector_space(
            self._reshaped, basis_vectors=data, n_dimensions=n_components
        )

    def components_sufficient_for_variance_ratio(
        self, variance_ratio: float
    ) -> np.ndarray:
        """Return the PCs account for given variance ratio."""
        n_components = self.number_of_components_sufficient_for_variance_ratio(
            variance_ratio
        )
        return self.components[:n_components]

    def number_of_components_sufficient_for_variance_ratio(
        self, variance_ratio: float
    ) -> int:

        """Return the PCs account for given variance ratio."""
        return self._index_of_variance_excess(variance_ratio)

    def _index_of_variance_excess(self, variance_ratio: float) -> int:
        # Find all indexes where the cumulative variance ratio exceeds the
        # given threshold.
        indexes = np.where(self.cumulative_variance_ratios >= variance_ratio)[0]
        # The first of these indexes is the index where the threshold is
        # exceeded. We want to include it as well and drop the rest.
        minimum_components_for_variance_ratio = indexes[0] + 1
        return minimum_components_for_variance_ratio


def _perform_varimax_rotation(
    matrix: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
):
    transposed = matrix.T
    n_row, n_col = transposed.shape
    rotation_matrix = np.eye(n_col)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(transposed, rotation_matrix)
        tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / n_row)
        u, s, v = np.linalg.svd(np.dot(matrix, comp_rot**3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    return np.dot(transposed, rotation_matrix).T


def _transform_data_into_vector_space(
    data: np.ndarray,
    basis_vectors: np.ndarray,
    n_dimensions: t.Optional[int] = None,
) -> np.ndarray:
    if n_dimensions is not None:
        basis_vectors = basis_vectors[:n_dimensions]
    centered = data - np.nanmean(data)
    return np.dot(centered, basis_vectors.T)


@lifetimes.utils.log_runtime
def spatio_temporal_principal_component_analysis(
    data: xr.DataArray,
    time_coordinate: str,
    latitude_coordinate: str = "latitude",
    x_coordinate: t.Optional[str] = None,
    y_coordinate: t.Optional[str] = None,
    variance_ratio: t.Optional[float] = None,
    pca_method: PCAMethod = decomposition.PCA,
    **kwargs,
) -> PCA:
    """Perform a spatio-temporal PCA.

    Parameters
    ----------
    data : xr.DataArray
        Spatial timeseries data.
    time_coordinate : str
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
    variance_ratio : float, optional
        Variance ratio threshold at which to drop the PCs.
    pca_method : sklearn.decomposition.PCA or IncrementalPCA, default=PCA
        Method to use for the PCA.
    kwargs
        Additional keyword arguments to pass to the PCA method.

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
    if variance_ratio is not None:
        if variance_ratio < 0.0 or variance_ratio > 1.0:
            raise ValueError("Variance ratio must be in the range [0;1]")
        pca = pca_method(n_components=variance_ratio, **kwargs)
    else:
        pca = pca_method(**kwargs)

    original_shape = data.shape
    timeseries = data[time_coordinate]
    data = lifetimes.utils.weight_by_latitudes(
        data=data,
        latitudes=latitude_coordinate,
        use_sqrt=True,
    )
    data = lifetimes.utils.reshape_spatio_temporal_xarray_data_array(
        data=data,
        time_coordinate=None,  # Set to None to avoid memory excess in function
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )

    result: PCAMethod = pca.fit(data)
    return PCA(
        pca=result,
        reshaped=data,
        original_shape=original_shape,
        time_series=timeseries,
    )
