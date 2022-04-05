import argparse
import functools
import itertools
import pathlib
import time
import typing as t

import distutils
import lifetimes

import mlflow


def pca_and_kmeans(
    path: t.Union[str, pathlib.Path],
    variance_ratio: float,
    n_clusters: int,
    use_varimax: bool,
) -> tuple[lifetimes.modes.methods.clustering.KMeans, list]:
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
    lifetimes.modes.methods.clustering.KMeans
        The trained KMeans model.
    list[lifetimes.modes.methods.Mode]
        All modes and their lifetime statistics.

    """
    ds = lifetimes.features.EcmwfIfsHresDataset(
        paths=[path],
        overlapping=False,
    )
    data = ds.as_xarray()["t"]

    modes = [lifetimes.modes.Modes(feature=data)]

    pca_partial_method = functools.partial(
        lifetimes.modes.methods.spatio_temporal_principal_component_analysis,
        variance_ratio=variance_ratio,
        time_coordinate="time",
        latitude_coordinate="latitude",
    )
    [pca] = lifetimes.modes.determine_modes(
        modes=modes, method=pca_partial_method
    )

    mlflow.log_metric("n_components", pca.n_components)

    clusters = lifetimes.modes.methods.find_principal_component_clusters(
        pca,
        use_varimax=use_varimax,
        n_clusters=n_clusters,
    )

    mlflow.log_metric("n_kmeans_iterations", clusters.model.n_iter_)

    cluster_lifetimes = lifetimes.modes.methods.determine_lifetimes_of_modes(
        modes=clusters.labels,
        time_coordinate="time",
    )
    return clusters, cluster_lifetimes


if __name__ == "__main__":

    def string_to_bool(s: str) -> bool:
        return bool(distutils.util.strtobool(s))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--variance-ratios", nargs="+", type=float)
    parser.add_argument("--n-clusters", nargs="+", type=int)
    parser.add_argument("--use-varimax", nargs="+", type=string_to_bool)
    args = parser.parse_args()

    parameters = itertools.product(
        args.variance_ratios, args.n_clusters, args.use_varimax
    )

    for variance_ratio, n_clusters, use_varimax in parameters:
        with mlflow.start_run():
            start = time.time()
            clusters, clusters_lifetimes = pca_and_kmeans(
                path=args.data,
                variance_ratio=variance_ratio,
                n_clusters=n_clusters,
                use_varimax=use_varimax,
            )
            end = time.time()
            duration = end - start

            mlflow.log_param("variance_ratio", variance_ratio)
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_param("use_varimax", use_varimax)
            mlflow.log_metric("duration", duration)
            mlflow.sklearn.log_model(clusters.model, "model")
            mlflow.log_artifact(args.data)
