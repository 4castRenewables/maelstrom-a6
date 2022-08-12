import typing as t

import lifetimes.utils as utils
import lifetimes.utils._types as _types
import numpy as np
import sklearn.decomposition as decomposition
import xarray as xr

PCAMethod = t.Union[decomposition.PCA, decomposition.IncrementalPCA]

PC_DIM = "component"


class PCA:
    """Wrapper for `sklearn.decomposition.PCA`."""

    def __init__(
        self,
        pca: PCAMethod,
        reshaped: np.ndarray,
        original_shape: tuple,
        time_series: xr.DataArray,
        x_coordinate: t.Optional[str] = None,
        y_coordinate: t.Optional[str] = None,
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
        x_coordinate : str, optional
            Name of the x coordinate in the original dataset.
        y_coordinate : str, optional
            Name of the y coordinate in the original dataset.

        """
        self._pca = pca
        self._original_reshaped = xr.DataArray(
            reshaped, dims=[*time_series.dims, "flattened_data"]
        )
        self._original_shape = original_shape
        self._time_series = time_series
        self._time_dims = time_series.dims
        self._spatial_dims = [
            x_coordinate or "longitude",
            y_coordinate or "latitude",
        ]

        self._components = xr.DataArray(
            self._pca.components_, dims=[PC_DIM, "entry"]
        )
        self._n_components: int = self._components.sizes[PC_DIM]
        self._explained_variance = xr.DataArray(
            self._pca.explained_variance_, dims=[PC_DIM]
        )
        self._explained_variance_ratio = xr.DataArray(
            self._pca.explained_variance_ratio_, dims=[PC_DIM]
        )
        self._cumulative_variance_ratio = xr.DataArray(
            np.cumsum(self._explained_variance_ratio), dims=[PC_DIM]
        )

    @property
    def timeseries(self) -> xr.DataArray:
        return self._time_series

    @property
    def components(self) -> xr.DataArray:
        """Return the principal components (EOFs)."""
        return self._components

    @property
    def n_components(self) -> int:
        """Return the number of principal components (EOFs)."""
        return self._n_components

    @property
    def explained_variance(self) -> xr.DataArray:
        """Return the corresponding eigenvalues."""
        return self._explained_variance

    @property
    def explained_variance_ratio(self) -> xr.DataArray:
        """Return the explained variance ratios."""
        return self._explained_variance_ratio

    @property
    def cumulative_variance_ratio(self) -> xr.DataArray:
        """Return the cumulative variance ratios."""
        return self._cumulative_variance_ratio

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

    @utils.log_runtime
    def transform(self, n_components: t.Optional[int] = None) -> xr.DataArray:
        """Transform the given data into the vector space of the PCs."""
        return self._transform(self.components, n_components=n_components)

    @utils.log_runtime
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
        self,
        components: xr.DataArray,
        n_components: t.Optional[int] = None,
    ) -> xr.DataArray:
        """Tansform data into PC space.

        See implementation of sklearn:
        https://github.com/scikit-learn/scikit-learn/blob/17df37aee774720212c27dbc34e6f1feef0e2482/sklearn/decomposition/_base.py#L100  # noqa

        """
        data = self._original_reshaped
        components = _select_components(components, n_components=n_components)

        if self._pca.mean_ is not None:
            data = data - self._pca.mean_

        transformed = utils.np_dot(data, components.T)

        if self._pca.whiten:
            transformed /= xr.ufuncs.sqrt(self.explained_variance)

        return transformed

    def inverse_transform(
        self,
        data: xr.DataArray,
        n_components: t.Optional[int] = None,
        in_original_shape: bool = True,
    ) -> xr.DataArray:
        """Transform data back to its original space.

        Parameters
        ----------
        data : xr.DataArray
            Data to transform.
        n_components : int, optional
            Number of PCs to use for the transformation.
            If `None`, all PCs are used.
        in_original_shape : bool, default=True
            Whether to return the transformed data in original shape.

        Notes
        -----
        See the implementation of `scikit-learn`:
        https://github.com/scikit-learn/scikit-learn/blob/6894a9be371683d4d61a861554c53f268c8771ca/sklearn/decomposition/_base.py#L128  # noqa

        """
        components = _select_components(
            self.components, n_components=n_components
        )
        if self._pca.whiten:
            explained_variance = _select_components(
                self.explained_variance, n_components=n_components
            )
            inverse = utils.np_dot(
                data,
                utils.np_dot(
                    np.sqrt(explained_variance.values[:, np.newaxis]),
                    components,
                ),
            )
            inverse += self._pca.mean_
        else:
            inverse = utils.np_dot(data, components) + self._pca.mean_

        if in_original_shape:
            return self._to_original_shape(
                inverse, includes_time_dimension=False
            )
        return inverse

    def _to_original_shape(
        self, data: xr.DataArray, includes_time_dimension: bool = True
    ) -> xr.DataArray:
        if includes_time_dimension:
            reshaped = data.data.reshape(self._original_shape)
            # The PCs are flipped along axis 1.
            return xr.DataArray(
                np.flip(reshaped, axis=1), dims=[PC_DIM, *self._spatial_dims]
            )
        reshaped = data.data.reshape(self._original_shape[1:])
        # The PCs are flipped along axis 0.
        return xr.DataArray(np.flip(reshaped, axis=0), dims=self._spatial_dims)

    def components_sufficient_for_variance_ratio(
        self, variance_ratio: float
    ) -> xr.DataArray:
        """Return the PCs account for given variance ratio."""
        n_components = self.number_of_components_sufficient_for_variance_ratio(
            variance_ratio
        )
        return _select_components(self.components, n_components=n_components)

    def number_of_components_sufficient_for_variance_ratio(
        self, variance_ratio: float
    ) -> int:

        """Return the PCs account for given variance ratio."""
        return self._index_of_variance_excess(variance_ratio)

    def _index_of_variance_excess(self, variance_ratio: float) -> int:
        # Find all indexes where the cumulative variance ratio exceeds the
        # given threshold.
        indexes = np.where(self.cumulative_variance_ratio >= variance_ratio)[0]
        # The first of these indexes is the index where the threshold is
        # exceeded. We want to include it as well and drop the rest.
        minimum_components_for_variance_ratio = indexes[0] + 1
        return minimum_components_for_variance_ratio


def _perform_varimax_rotation(
    matrix: xr.DataArray,
    tolerance: float = 1e-6,
    max_iter: int = 100,
) -> xr.DataArray:
    transposed = matrix.data.T
    n_row, n_col = transposed.shape
    rotation_matrix = np.eye(n_col)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(transposed, rotation_matrix)
        tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / n_row)
        u, s, v = np.linalg.svd(np.dot(matrix.data, comp_rot**3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tolerance):
            break
        var = var_new

    return xr.DataArray(np.dot(transposed, rotation_matrix).T, dims=matrix.dims)


def _select_components(
    data: xr.DataArray, n_components: t.Optional[int]
) -> xr.DataArray:
    if n_components is None:
        return data
    return data.sel({PC_DIM: slice(n_components)})


@utils.log_runtime
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
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )


def _reshape_data(
    data: _types.Data,
    time_coordinate: str,
    latitude_coordinate: str,
    x_coordinate: t.Optional[str],
    y_coordinate: t.Optional[str],
) -> tuple[tuple, xr.DataArray, np.ndarray]:
    timeseries = data[time_coordinate]
    original_shape = utils.get_xarray_data_shape(data)
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
        original_shape,
        timeseries,
        data,
    )
