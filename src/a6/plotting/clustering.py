import typing as t

import a6.modes.methods.clustering as clustering
import a6.plotting.pca as pca
import matplotlib.pyplot as plt
import seaborn.palettes

Plot = tuple[plt.Figure, plt.Axes]


class AxesFactory(t.Protocol):
    def __call__(self, axis: plt.Axes, *args, **kwargs) -> plt.Axes:
        ...


def plot_first_three_components_timeseries_clusters(
    clusters: clustering.ClusterAlgorithm, display: bool = True
) -> Plot:
    """Plot the first three PCs and the clusters."""
    fig, ax = pca.plot_first_three_components_timeseries(
        pca=clusters.pca,
        colors=_create_colors(clusters),
        display=display,
    )

    if isinstance(clusters, clustering.KMeans):
        x_centers, y_centers, z_centers = clusters.centers.T[:3]
        ax.scatter(x_centers, y_centers, z_centers, c="red", marker="+")

    return fig, ax


def plot_condensed_tree(
    clusters: clustering.HDBSCAN,
    highlight_selected_clusters: bool = True,
    **kwargs
) -> Plot:
    """Plot the condensed tree of an HDBSCAN model.

    Parameters
    ----------
    clusters : clustering.HDBSCAN
        The cluster model.
    highlight_selected_clusters : bool, default=True
        Whether to highlight the clusters that were selected by the algorithm
        as the main clusters.

    Raises
    ------
    ValueError
        If the given algorithm is not a hierarchical clustering algorithm.

    Returns
    -------
    plt.Figure
    plt.Axes

    """
    _raise_if_not_hierarchical_clustering_method(clusters)
    return _create_plot(
        ax_factory=clusters.model.condensed_tree_.plot,
        select_clusters=highlight_selected_clusters,
        selection_palette=_create_colormap(clusters),
        **kwargs,
    )


def plot_single_linkage_tree(
    clusters: clustering.HDBSCAN,
) -> Plot:
    """Plot the single-linkage tree of an HDBSCAN model.

    Parameters
    ----------
    clusters : clustering.HDBSCAN
        The cluster model.

    Raises
    ------
    ValueError
        If the given algorithm is not a hierarchical clustering algorithm.

    Returns
    -------
    plt.Figure
    plt.Axes

    """
    _raise_if_not_hierarchical_clustering_method(clusters)
    return _create_plot(ax_factory=clusters.model.single_linkage_tree_.plot)


def _create_plot(ax_factory: AxesFactory, **kwargs) -> Plot:
    fig, ax = plt.subplots()
    ax_factory(axis=ax, **kwargs)
    return fig, ax


def _create_colors(clusters: clustering.ClusterAlgorithm) -> list[str]:
    cmap = _create_colormap(clusters)
    return [cmap[i] if i != -1 else "black" for i in clusters.labels.values]


def _raise_if_not_hierarchical_clustering_method(clusters: t.Any):
    if not isinstance(clusters, clustering.HDBSCAN):
        raise ValueError(
            "Condensed tree can only be plotted for a hierarchical "
            "clustering algorithm"
        )


def _create_colormap(
    clusters: clustering.ClusterAlgorithm,
) -> seaborn.palettes._ColorPalette:
    n_labels = clusters.labels.values.max() + 1
    return seaborn.color_palette(n_colors=n_labels)
