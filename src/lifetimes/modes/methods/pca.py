import typing as t
import functools
import lifetimes.utils
import numpy as np
import xarray as xr
import sklearn.decomposition as decomposition

import lifetimes.utils
import lifetimes.coordinate_transformations as transformations

PCAMethod = t.Union[t.Type[decomposition.PCA], t.Type[decomposition.IncrementalPCA]]


class PCA(transformations.InvertibleTransformation):
    """Wrapper for `sklearn.decomposition.PCA`."""

    def __init__(
        self,
        transformation_as_dataset: xr.Dataset,
        pca: decomposition.PCA,
        data: xr.DataArray,
        reshaped: np.ndarray,
        time_coordinate: str,
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
        super().__init__(
            transformation_as_dataset=transformation_as_dataset, is_semi_orthogonal=True
        )
        self._data: xr.Dataset = data.copy().to_dataset(name="data")
        self._time_coordinate = time_coordinate
        self._time_series = data.coords[time_coordinate]
        self._reshaped = reshaped.copy()
        self._pca = pca
        self._data_variable_name = list(self._data.keys())[0]
        self.mean = transformation_as_dataset["mean"].to_dataset(
            name=self._data_variable_name
        )

    @property
    def timeseries(self) -> xr.DataArray:
        return self._time_series

    @property
    def data_variable_name(self):
        return self._data_variable_name

    @property
    def components(self) -> np.ndarray:
        """Return the principal components (EOFs)."""
        n_eigenvectors = len(self.eigenvalues)
        return self.matrix.values.reshape(n_eigenvectors, -1).copy()

    @property
    def n_components(self) -> int:
        """Return the number of principal components (EOFs)."""
        return self.components.shape[0]

    @property
    def components_in_original_shape(self) -> np.ndarray:
        """Return the principal components (EOFs)."""
        return self.matrix.values

    @property
    def loadings(self) -> np.ndarray:
        """Return the loadings of each PC."""
        return (
            self.components.T * np.sqrt(self.eigenvalues)
        ).T

    @property
    def variance_ratios(self) -> np.ndarray:
        """Return the explained variance ratios."""
        return self._pca.explained_variance_ratio_

    @property
    def cumulative_variance_ratios(self) -> np.ndarray:
        """Return the cumulative variance ratios."""
        return np.cumsum(self.variance_ratios)

    @functools.lru_cache
    def transform(
        self,
        data: t.Optional[xr.Dataset] = None,
        n_components: t.Optional[int] = None,
        target_variable: t.Optional[str] = None,
    ) -> xr.Dataset:
        if data is None:
            data = self._data
        if target_variable is None:
            target_variable = self._data_variable_name
        return super(PCA, self).transform(
            data=data, n_dimensions=n_components, target_variable=target_variable
        ) - super(PCA, self).transform(
            data=self.mean, n_dimensions=n_components, target_variable=target_variable
        )

    @functools.lru_cache
    def inverse_transform(
        self,
        data: xr.Dataset,
        n_components: t.Optional[int] = None,
        target_variable: t.Optional[str] = None,
    ):
        if target_variable is None:
            target_variable = list(self._data.keys())[0]
        return (
            super(PCA, self).inverse_transform(
                data=data, n_dimensions=n_components, target_variable=target_variable
            )
            + self.mean
        )

    @functools.lru_cache
    def components_sufficient_for_variance_ratio(
        self, variance_ratio: float
    ) -> np.ndarray:
        """Return the PCs account for given variance ratio."""
        n_components = self.number_of_components_sufficient_for_variance_ratio(
            variance_ratio
        )
        return self.components[:n_components]

    @functools.lru_cache
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

    data = lifetimes.utils.weight_by_latitudes(
        data=data,
        latitudes=latitude_coordinate,
        use_sqrt=True,
    )
    reshaped = lifetimes.utils.reshape_spatio_temporal_xarray_data_array(
        data=data,
        time_coordinate=None,  # Set to None to avoid memory excess in the function.
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )

    result: decomposition.PCA = pca.fit(reshaped)

    result_as_dataset = _pca_to_dataset_representation(
        pca, data, x_coordinate, y_coordinate, time_coordinate
    )
    return PCA(
        transformation_as_dataset=result_as_dataset,
        pca=result,
        data=data,
        reshaped=reshaped,
        time_coordinate=time_coordinate,
    )


def _pca_to_dataset_representation(
    pca: decomposition.PCA,
    input_dataset: xr.Dataset,
    x_coordinate: str,
    y_coordinate: str,
    time_coordinate: str,
) -> xr.Dataset:
    """
    Represent PCA as xr.Dataset including eigenvectors and eigenvalues.

    Parameters
    ----------
    pca: Trained PCA instance.
    input_dataset: Dataset used for training.

    Returns
    -------
    Dataset representation of PCA.
    """
    x_coordinates = input_dataset.coords[x_coordinate].values
    y_coordinates = input_dataset.coords[y_coordinate].values
    training_dimensions = {x_coordinate: x_coordinates, y_coordinate: y_coordinates}
    training_dimension_shapes = [x_coordinates.shape[0], y_coordinates.shape[0]]
    n_observations = input_dataset.coords[time_coordinate].values.shape[0]
    coords = {
        "eigenvector_number": list(range(n_observations)),
        x_coordinate: x_coordinates,
        y_coordinate: y_coordinates,
    }
    return xr.Dataset(
        data_vars={
            "transformation_matrix": (
                ["eigenvector_number", *training_dimensions],
                pca.components_.reshape(-1, *training_dimension_shapes),
            ),
            "mean": (
                training_dimensions,
                pca.mean_.reshape(*training_dimension_shapes),
            ),
            "eigenvalues": (["eigenvector_number"], pca.explained_variance_),
        },
        coords=coords,
    )
