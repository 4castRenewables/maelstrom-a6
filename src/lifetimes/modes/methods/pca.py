from typing import Optional

import lifetimes.utils
import xarray as xr
from sklearn.decomposition import PCA


def spatio_temporal_principal_component_analysis(
    data: xr.DataArray,
    time_coordinate: str,
    x_coordinate: Optional[str] = None,
    y_coordinate: Optional[str] = None,
    variance_ratio: Optional[float] = None,
) -> PCA:
    """Perform a spatio-temporal PCA.

    Parameters
    ----------
    data : xr.DataArray
        Spatial timeseries data.
    time_coordinate : str
        Name of the time coordinate.
        This is required to reshape the data for the PCA.
    x_coordinate : str, optional
        Name of the x-coordinate of the grid.
        If `None`, it is important to always use the data with the exact same
        shape.
    y_coordinate : str, optional
        Name of the y-coordinate of the grid.
        If `None`, it is important to always use the data with the exact same
        shape.
    variance_ratio : float, optional
        Variance ratio threshold at which to drop the PCs.

    Returns
    -------
    xr.DataArray
        Contains the PCs.

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
        pca = PCA(n_components=variance_ratio)
    else:
        pca = PCA()

    reshaped = lifetimes.utils.reshape_spatio_temporal_xarray_data_array(
        data=data,
        time_coordinate=time_coordinate,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )
    pcs = pca.fit(reshaped)
    return pcs
