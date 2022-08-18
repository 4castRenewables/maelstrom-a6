import contextlib
import dataclasses
import itertools
import typing as t

import hdbscan
import lifetimes.modes.methods as methods
import lifetimes.modes.methods.clustering as clustering
import lifetimes.modes.methods.pca as _pca

import mlflow


@dataclasses.dataclass
class HyperParameters:
    """Hyperparameters for the hyperparameter study."""

    n_components_start: int
    min_cluster_size_start: int
    n_components_end: t.Optional[int] = None
    min_cluster_size_end: t.Optional[int] = None

    def to_range(self) -> t.Iterator:
        """Return as a range to use in a for loop."""
        if self.n_components_end is None:
            n_components_end = self.n_components_start
        else:
            n_components_end = self.n_components_end
        n_components_range = range(
            self.n_components_start, n_components_end + 1
        )

        if self.min_cluster_size_end is None:
            min_cluster_size_end = self.min_cluster_size_start
        else:
            min_cluster_size_end = self.min_cluster_size_end
        min_cluster_size_range = range(
            self.min_cluster_size_start, min_cluster_size_end + 1
        )

        return itertools.product(n_components_range, min_cluster_size_range)


def perform_hdbscan_hyperparameter_study(
    pca: _pca.PCA, hyperparameters: HyperParameters, log_to_mantik: bool = True
) -> list[clustering.ClusterAlgorithm]:
    """Do a hyperparameter study for the HDBSCAN clustering."""
    context = mlflow.start_run if log_to_mantik else contextlib.nullcontext
    clusters = []

    for (
        n_components,
        min_cluster_size,
    ) in hyperparameters.to_range():
        with context():
            algorithm = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            clusters_temp = methods.find_pc_space_clusters(
                algorithm=algorithm,
                pca=pca,
                n_components=n_components,
                use_varimax=False,
            )
            clusters.append(clusters_temp)

            if log_to_mantik:
                mlflow.log_param("n_components", n_components)
                mlflow.log_param("hdbscan_min_cluster_size", min_cluster_size)
                mlflow.log_metric("n_clusters", clusters_temp.n_clusters)
    return clusters
