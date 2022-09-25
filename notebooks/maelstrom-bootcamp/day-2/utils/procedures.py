import hdbscan
import numpy as np
import sklearn.decomposition as decomposition
import xarray as xr


def perform_pca_and_transform_into_pc_space(
    data: xr.DataArray, n_components: int
) -> np.ndarray:
    """Perform a PCA on the given dataset and directly transform the data into
    PC space.

    Parameters
    ----------
    data : xr.DataArray
    n_components : int
        Number of PCs to use for the transformation.

    Returns
    -------
    np.ndarray
        The transformed, `n_components`-dimensional dataset.

    """
    matrix = reshape_grid_time_series(data)
    pca = decomposition.PCA(n_components=n_components).fit(matrix)
    return pca.transform(matrix)


def reshape_grid_time_series(data: xr.DataArray) -> np.ndarray:
    """Reshape the given data array by concatenating rows."""
    return np.array([step.values.flatten() for step in data])


def restore_original_grid_shape(
    original: xr.Dataset, reshape: np.array
) -> xr.DataArray:
    """Restore the original shape of the grid.

    Here, we return a xr.DataArray to directly use its `plot()`
    to plot the result.

    The PCs are flipped, which is why we flip them along the
    x-axis.

    """
    var = list(original.data_vars)[0]
    _, x, y = original[var].isel(time=0).values.shape
    return xr.DataArray(np.flip(reshape.reshape((x, y)), axis=0))


def perform_hdbscan(
    data: xr.DataArray, n_components: int, min_cluster_size: int
) -> hdbscan.HDBSCAN:
    """Perform a clustering with HDBSCAN on the given data."""
    transformed = perform_pca_and_transform_into_pc_space(
        data,
        n_components=n_components,
    )
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(transformed)
