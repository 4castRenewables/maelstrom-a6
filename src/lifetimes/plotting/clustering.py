import lifetimes.modes.methods.clustering as clustering
import lifetimes.plotting.pca as pca
import matplotlib.pyplot as plt


def plot_first_three_components_timeseries_clusters(
    clusters: clustering.ClusterAlgorithm, display: bool = True
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the first three PCs and the clusters."""
    fig, ax = pca.plot_first_three_components_timeseries(
        pca=clusters.pca,
        colors=clusters.labels,
        display=display,
    )

    if isinstance(clusters, clustering.KMeans):
        x_centers, y_centers, z_centers = clusters.centers.T[:3]
        ax.scatter(x_centers, y_centers, z_centers, c="red", marker="+")

    return fig, ax


def plot_condensed_tree(
    clusters: clustering.HDBSCAN, highlight_selected_clusters: bool = True
) -> plt.Axes:
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
    matplotlib.Axes

    """
    if not isinstance(clusters, clustering.HDBSCAN):
        raise ValueError(
            "Condensed tree can only be plotted for a hierarchical "
            "clustering algorithm"
        )
    return clusters.model.condensed_tree_.plot(
        select_clusters=highlight_selected_clusters
    )
