import matplotlib.pyplot as plt

import lifetimes.modes.methods.clustering as clustering

import lifetimes.plotting.pca as pca


def plot_first_three_components_timeseries_clusters(clusters: clustering.ClusterAlgorithm, display: bool = True) -> tuple[plt.Figure, plt.Axes]:
    """Plot the first three PCs and the clusters."""
    fig, ax = pca.plot_first_three_components_timeseries(
        pca=clusters.pca,
        colors=clusters.labels,
        display=display,
    )
    x_centers, y_centers, z_centers = clusters.centers.T[:3]
    ax.scatter(x_centers, y_centers, z_centers, c="red", marker="+")

    return fig, ax
