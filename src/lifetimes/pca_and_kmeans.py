import functools
import logging
import pathlib
import time
import typing as t

import lifetimes.features
import lifetimes.modes

logger = logging.getLogger(__name__)


def pca_and_kmeans(
    path: t.Union[str, pathlib.Path],
    variance_ratio: float,
    n_clusters: int,
    use_varimax: bool,
) -> list:
    """Run PCA and K-means on ECMWF IFS HRES data.

    Parameters
    ----------
    path : str or pathlib.Path
        List to the file containing the data.
    variance_ratio : float
        Variance ratio to cover with the PCs for the clustering.
    n_clusters : int
        Number of clusters to separate the data into.
    use_varimax : bool
        Whether to use varimax rotation for the PCs before clustering.

    Returns
    -------
    list[lifetimes.modes.methods.Mode]
        All modes and their lifetime statistics.

    """
    start_single_job = time.time()

    ds = lifetimes.features.EcmwfIfsHresDataset(
        paths=[path],
        overlapping=False,
    )
    data = ds.as_xarray()["t"]

    modes = [lifetimes.modes.Modes(feature=data)]

    pca_partial_method = functools.partial(
        lifetimes.modes.methods.spatio_temporal_pca,
        variance_ratio=variance_ratio,
        time_coordinate="time",
        latitude_coordinate="latitude",
    )
    [pca] = lifetimes.modes.determine_modes(
        modes=modes, method=pca_partial_method
    )

    clusters = lifetimes.modes.methods.find_pc_space_clusters(
        pca,
        use_varimax=use_varimax,
        n_clusters=n_clusters,
    )

    cluster_lifetimes = lifetimes.modes.methods.determine_lifetimes_of_modes(
        modes=clusters.labels,
        time_coordinate="time",
    )

    logger.info(
        (
            "PCA + K-means runtime with variance_ratio %s and n_clusters %s "
            "(use_varimax is %s): %s seconds"
        ),
        variance_ratio,
        n_clusters,
        use_varimax,
        time.time() - start_single_job,
    )
    return cluster_lifetimes
