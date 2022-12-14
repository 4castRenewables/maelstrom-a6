import dataclasses
import itertools
from collections.abc import Iterator

import a6.cli.options.config as _config
import a6.types as types


@dataclasses.dataclass
class HyperParameters:
    """Hyperparameters for the hyperparameter study.

    Parameters
    ----------
    cluster_arg : str
        Name of the argument to the clustering algorithm
    n_components_start : int, default=1
        Value for `n_components` (PCA) where to start.
    n_components_end : int, optional
        Value for `n_components` (PCA) where to stop.
    cluster_start : int, default=2
        Value for `min_cluster_size` (HDBSCAN) where to start.
    cluster_end : int, optional
        Value for `min_cluster_size` (HDBSCAN) where to stop.

    """

    cluster_arg: str
    n_components_start: int = 1
    n_components_end: int | None = None
    cluster_start: int = 2
    cluster_end: int | None = None

    @classmethod
    def from_config(cls, config: _config.Config) -> "HyperParameters":
        """Create from CLI config."""
        parameters = config.parameters

        def read_optional(name: str) -> int | None:
            return parameters[name] if name in parameters else None

        return cls(
            cluster_arg=parameters["cluster_arg"],
            n_components_start=parameters["n_components_start"],
            n_components_end=read_optional("n_components_end"),
            cluster_start=parameters["cluster_start"],
            cluster_end=read_optional("cluster_end"),
        )

    @property
    def n_components_max(self) -> int:
        """Return the maximum amount of PCs used."""
        return self.n_components_end or self.n_components_start

    @property
    def _n_components_range(self) -> Iterator:
        """Return the `range` for `n_components`."""

        if self.n_components_end is None:
            n_components_end = self.n_components_start
        else:
            n_components_end = self.n_components_end
        return range(self.n_components_start, n_components_end + 1)

    @property
    def _clustering_range(self) -> Iterator:
        """Return the `range` for `min_cluster_size`."""
        if self.cluster_end is None:
            cluster_end = self.cluster_start
        else:
            cluster_end = self.cluster_end
        return range(self.cluster_start, cluster_end + 1)

    def to_range(self) -> Iterator:
        """Return as a range to use in a for loop."""
        return itertools.product(
            self._n_components_range, self._clustering_range
        )

    def apply(
        self, algorithm: type[types.ClusterAlgorithm], cluster_param: int
    ) -> types.ClusterAlgorithm:
        """Apply to a given algorithm."""
        return algorithm(**{self.cluster_arg: cluster_param})
