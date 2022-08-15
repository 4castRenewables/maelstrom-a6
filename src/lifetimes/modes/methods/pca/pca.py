import typing as t

import lifetimes.modes.methods.pca._reshape as _reshape
import lifetimes.utils as utils
import numpy as np
import sklearn.decomposition as decomposition
import xarray as xr

PCAMethod = t.Union[decomposition.PCA, decomposition.IncrementalPCA]

PC_DIM = "component"
PC_VALUES_DIM = "entry"


class PCA:
    """Wrapper for `sklearn.decomposition.PCA`."""

    def __init__(
        self,
        pca: PCAMethod,
        reshaped: np.ndarray,
        dimensions: utils.Dimensions,
    ):
        """Wrap `sklearn.decomposition.PCA`.

        Parameters
        ----------
        pca : sklearn.decomposition.PCA
        reshaped : np.ndarray
            The reshaped, original data used for PCA.
        dimensions : lifetimes.utils.dimensions.Dimensions
            Original shape of the data the PCA was performed on.

        """
        self._pca = pca
        self._original_reshaped = xr.DataArray(
            reshaped, dims=[dimensions.time.name, "flattened_data"]
        )
        self._dimensions = dimensions
        self.reshaper = _reshape.Reshaper(dimensions)

        self._components = xr.DataArray(
            self._pca.components_, dims=[PC_DIM, PC_VALUES_DIM]
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
        return self._dimensions.time.values

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
    def components_in_original_shape(self) -> xr.Dataset:
        """Return the principal components (EOFs)."""
        return self.reshaper(self.components)

    @property
    def components_varimax_rotated(self) -> xr.DataArray:
        """Return the principal components (EOFs)."""
        return _perform_varimax_rotation(self.components)

    @property
    def components_varimax_rotated_in_original_shape(self) -> xr.Dataset:
        """Return the principal components (EOFs)."""
        return self.reshaper(self.components_varimax_rotated)

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

    @utils.log_runtime
    def inverse_transform(
        self,
        data: xr.DataArray,
        n_components: t.Optional[int] = None,
        in_original_shape: bool = True,
    ) -> xr.Dataset:
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
                    np.sqrt(explained_variance.data[:, np.newaxis]),
                    components,
                ),
            )
            inverse += self._pca.mean_
        else:
            inverse = utils.np_dot(data, components) + self._pca.mean_

        if in_original_shape:
            # If the given data is unlabeled, the name of the dimension is
            # `dim_0`.
            return self.reshaper(
                inverse,
                includes_time_dimension=False,
                rename_dim_0=True,
            )
        name = inverse.name or "_".join(self._dimensions.variable_names)
        return xr.Dataset(data_vars={name: inverse}, coords=inverse.coords)

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
