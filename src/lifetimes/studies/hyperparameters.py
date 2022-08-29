import dataclasses
import itertools
import typing as t


@dataclasses.dataclass
class HyperParameters:
    """Hyperparameters for the hyperparameter study.

    Parameters
    ----------
    n_components_start : int, default=1
        Value for `n_components` (PCA) where to start.
    n_components_end : int, optional
        Value for `n_components` (PCA) where to stop.
    min_cluster_size_start : int, default=2
        Value for `min_cluster_size` (HDBSCAN) where to start.
    min_cluster_size_end : int, optional
        Value for `min_cluster_size` (HDBSCAN) where to stop.

    """

    n_components_start: int = 1
    n_components_end: t.Optional[int] = None
    min_cluster_size_start: int = 2
    min_cluster_size_end: t.Optional[int] = None

    @property
    def n_components_max(self) -> int:
        """Return the maximum amount of PCs used."""
        return self.n_components_end or self.n_components_start

    @property
    def n_components_range(self) -> t.Iterator:
        """Return the `range` for `n_components`."""

        if self.n_components_end is None:
            n_components_end = self.n_components_start
        else:
            n_components_end = self.n_components_end
        return range(self.n_components_start, n_components_end + 1)

    @property
    def min_cluster_size_range(self) -> t.Iterator:
        """Return the `range` for `min_cluster_size`."""
        if self.min_cluster_size_end is None:
            min_cluster_size_end = self.min_cluster_size_start
        else:
            min_cluster_size_end = self.min_cluster_size_end
        return range(self.min_cluster_size_start, min_cluster_size_end + 1)

    def to_range(self) -> t.Iterator:
        """Return as a range to use in a for loop."""
        return itertools.product(
            self.n_components_range, self.min_cluster_size_range
        )
