import typing as t

import lifetimes.utils
import lifetimes.utils._types as _types
import numpy as np
import sklearn.decomposition as decomposition
import xarray as xr

PCAMethod = t.Union[decomposition.PCA, decomposition.IncrementalPCA]


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
    def components(self) -> xr.DataArray:
        """Return the principal components (EOFs)."""
        return xr.DataArray(self._pca.components_, dims=["component", "entry"])

    @property
    def n_components(self) -> int:
        """Return the number of principal components (EOFs)."""
        return self._pca.components_.shape[0]

    @property
    def components_in_original_shape(self) -> xr.DataArray:
        """Return the principal components (EOFs)."""
        return self._to_original_shape(self.components)

    @property
    def components_varimax_rotated(self) -> xr.DataArray:
        """Return the principal components (EOFs)."""
        return _perform_varimax_rotation(self.components)

    @property
    def components_varimax_rotated_in_original_shape(self) -> xr.DataArray:
        """Return the principal components (EOFs)."""
        return self._to_original_shape(self.components_varimax_rotated)

    def _to_original_shape(
        self, data: xr.DataArray, includes_time_dimension: bool = True
    ) -> xr.DataArray:
        if includes_time_dimension:
            reshaped = data.values.reshape(self._original_shape)
            # The PCs are flipped along axis 1.
            return xr.DataArray(
                np.flip(reshaped, axis=1), dims=["component", "lat", "lon"]
            )
        reshaped = data.values.reshape(self._original_shape[1:])
        # The PCs are flipped along axis 0.
        return xr.DataArray(np.flip(reshaped, axis=0), dims=["lat", "lon"])

    @property
    def eigenvalues(self) -> xr.DataArray:
        """Return the corresponding eigenvalues."""
        return xr.DataArray(self._pca.explained_variance_)

    @property
    def loadings(self) -> np.ndarray:
        """Return the loadings of each PC."""
        return (self._pca.components_.T * np.sqrt(self.eigenvalues.values)).T

    @property
    def variance_ratios(self) -> xr.DataArray:
        """Return the explained variance ratios."""
        return xr.DataArray(self._pca.explained_variance_ratio_)

    @property
    def cumulative_variance_ratios(self) -> xr.DataArray:
        """Return the cumulative variance ratios."""
        return xr.DataArray(np.cumsum(self.variance_ratios))

    @lifetimes.utils.log_runtime
    def transform(self, n_components: t.Optional[int] = None) -> xr.DataArray:
        """Transform the given data into the vector space of the PCs."""
        return self._transform(self.components, n_components=n_components)

    @lifetimes.utils.log_runtime
    def transform_with_varimax_rotation(
        self, n_components: t.Optional[int] = None
    ) -> xr.DataArray:
        """Transform the given data into the
        vector space of the varimax-rotated PCs."""
        # TODO: Do research on whether the varimax rotation has to be performed
        # on the reduced number of PCs to achieve `n_components` or on all PCs
        # to then select `n_copoments` of the varimax-rotated PCs. Currently,
        # the former is implemented. Thus, varimax rotation is very efficient.
        return self._transform(
            self.components_varimax_rotated, n_components=n_components
        )

    def _transform(
        self, components: xr.DataArray, n_components: int
    ) -> xr.DataArray:
        return _transform_data_into_vector_space(
            self._reshaped, basis_vectors=components, n_dimensions=n_components
        )

    def inverse_transform(
        self, data: xr.DataArray, n_components: t.Optional[int] = None
    ) -> xr.DataArray:
        """Transform data back to its original space.

        Parameters
        ----------
        data : np.ndarray
            Data to transform.
        n_components : int, optional
            Number of PCs to use for the transformation.
            If `None`, all PCs are used.

        Notes
        -----
        See the implementation of `scikit-learn`:
        https://github.com/scikit-learn/scikit-learn/blob/6894a9be371683d4d61a861554c53f268c8771ca/sklearn/decomposition/_base.py#L128  # noqa

        """
        components = (
            self.components[:n_components]
            if n_components is not None
            else self.components
        )
        if self._pca.whiten:
            eigenvalues = (
                self.eigenvalues[:n_components]
                if components is not None
                else self.eigenvalues
            )
            inverse = (
                np.dot(
                    data,
                    np.sqrt(eigenvalues[:, np.newaxis]) * components,
                )
                + self._pca.mean_
            )
        else:
            inverse = np.dot(data, components) + self._pca.mean_
        return self._to_original_shape(
            xr.DataArray(inverse), includes_time_dimension=False
        )

    def components_sufficient_for_variance_ratio(
        self, variance_ratio: float
    ) -> xr.DataArray:
        """Return the PCs account for given variance ratio."""
        n_components = self.number_of_components_sufficient_for_variance_ratio(
            variance_ratio
        )
        return xr.DataArray(self.components.values[:n_components])

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
    matrix: xr.DataArray,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> xr.DataArray:
    transposed = matrix.values.T
    n_row, n_col = transposed.shape
    rotation_matrix = np.eye(n_col)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(transposed, rotation_matrix)
        tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / n_row)
        u, s, v = np.linalg.svd(np.dot(matrix.values, comp_rot**3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    return xr.DataArray(np.dot(transposed, rotation_matrix).T, dims=matrix.dims)


def _transform_data_into_vector_space(
    data: np.ndarray,
    basis_vectors: xr.DataArray,
    n_dimensions: t.Optional[int] = None,
) -> xr.DataArray:
    if n_dimensions is not None:
        basis_vectors = basis_vectors.values[:n_dimensions]
    centered = data - np.nanmean(data)
    return xr.DataArray(np.dot(centered, basis_vectors.T))


@lifetimes.utils.log_runtime
def spatio_temporal_pca(
    data: _types.Data,
    time_coordinate: str = "time",
    latitude_coordinate: str = "latitude",
    x_coordinate: t.Optional[str] = None,
    y_coordinate: t.Optional[str] = None,
    variance_ratio: t.Optional[float] = None,
    pca_method: t.Type[PCAMethod] = decomposition.PCA,
    **kwargs,
) -> PCA:
    """Perform a spatio-temporal PCA.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
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

    (original_shape, timeseries, data) = _reshape_data(
        data=data,
        time_coordinate=time_coordinate,
        latitude_coordinate=latitude_coordinate,
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


def _reshape_data(
    data: _types.Data,
    time_coordinate: str,
    latitude_coordinate: str,
    x_coordinate: t.Optional[str],
    y_coordinate: t.Optional[str],
) -> tuple[tuple, xr.DataArray, np.ndarray]:
    timeseries = data[time_coordinate]
    original_shape = lifetimes.utils.get_xarray_data_shape(data)
    data = lifetimes.utils.weight_by_latitudes(
        data=data,
        latitudes=latitude_coordinate,
        use_sqrt=True,
    )
    data = lifetimes.utils.reshape_spatio_temporal_xarray_data(
        data=data,
        time_coordinate=None,  # Set to None to avoid memory excess in function
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )
    return (
        original_shape,
        timeseries,
        data,
    )
