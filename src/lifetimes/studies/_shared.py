import contextlib
import typing as t

import lifetimes.modes.methods.clustering as clustering
import lifetimes.modes.methods.pca as _pca
import lifetimes.plotting as plotting
import lifetimes.utils as utils

import mlflow


def get_contexts(log_to_mantik: bool) -> contextlib.contextmanager:
    return get_mlflow_context(log_to_mantik), get_logs_context(log_to_mantik)


def get_mlflow_context(log_to_mantik: bool) -> contextlib.contextmanager:
    return mlflow.start_run if log_to_mantik else _nullcontext


def get_logs_context(log_to_mantik: bool) -> contextlib.contextmanager:
    return utils.log_logs_as_file if log_to_mantik else _nullcontext


@contextlib.contextmanager
def _nullcontext(*args, **kwargs) -> None:
    yield


def log_to_mantik(
    pca: _pca.PCA,
    clusters: clustering.ClusterAlgorithm,
    n_components: int,
    min_cluster_size: t.Optional[int] = None,
    use_varimax: t.Optional[bool] = None,
) -> None:
    plotting.create_plots_and_log_to_mantik(
        pca=pca,
        clusters=clusters,
    )
    mlflow.log_param("n_components", n_components)
    mlflow.log_metric("n_clusters", clusters.n_clusters)

    if min_cluster_size is not None:
        mlflow.log_param("hdbscan_min_cluster_size", min_cluster_size)

    if use_varimax is not None:
        mlflow.log_param("use_varimax", use_varimax)
