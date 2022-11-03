import contextlib

import a6.modes.methods.clustering as clustering
import a6.modes.methods.pca as _pca
import a6.plotting as plotting
import a6.utils as utils

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
    use_varimax: bool | None = None,
    **kwargs,
) -> None:
    plotting.create_plots_and_log_to_mantik(
        pca=pca,
        clusters=clusters,
    )
    mlflow.log_param("n_components", n_components)
    mlflow.log_metric("n_clusters", clusters.n_clusters)

    if kwargs:
        mlflow.log_params(kwargs)

    if use_varimax is not None:
        mlflow.log_param("use_varimax", use_varimax)
