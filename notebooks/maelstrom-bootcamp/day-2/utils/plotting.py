from typing import Optional
from typing import Union

import hdbscan
import matplotlib.animation as animation
import matplotlib.collections as collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import xarray as xr


def create_heatmap_plot(df: pd.DataFrame):
    """Create a heatmap plot of a 2-D `pd.DataFrame`."""
    return seaborn.heatmap(df, annot=True, cmap="coolwarm")


def create_production_and_forecast_comparison_plot(
    real: xr.DataArray,
    forecast: np.array,
    time_coordinate: str = "time",
    production_column: str = "production",
) -> plt.Figure:
    """Plot the real power production and the forecast.

    Parameters
    ----------
    real : xr.DataArray
        The real production data.
    forecast : np.array
        The forecast.
    time_coordinate : str, default="time"
        Name of the time coordinate.
    production_column : str, default="production"
        Name of the column in `real` that contains the production data.

    Returns
    -------
    plt.Figure

    """
    fig = plt.figure()

    plt.plot(real[time_coordinate], real[production_column], label="real")
    plt.plot(real[time_coordinate], forecast, label="forecast")
    plt.title("Forecast and real power production")
    plt.legend()

    return fig


def create_scree_test_plot(
    pca: decomposition.PCA,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create scree test plot of a PCA."""
    # Create the figure and axis.
    fig, ax1 = plt.subplots()

    # x ticks is the number of PCs.
    x = range(1, pca.n_components_ + 1)

    # Plot explained variance ratio on left y-axis.
    ax1.scatter(x, pca.explained_variance_ratio_, color="tab:blue")
    ax1.set_xlabel("n_components")
    ax1.set_ylabel("explained variance ratio", color="tab:blue")

    # Plot cumulative explained variance ratio on right y-axis.
    ax2 = ax1.twinx()
    ax2.scatter(x, np.cumsum(pca.explained_variance_ratio_), color="tab:red")
    ax2.set_ylabel("cumulative explained variance ratio", color="tab:red")

    return fig, (ax1, ax2)


def create_3d_scatter_plot(
    data: np.ndarray,
    colors: Optional[list[int]] = None,
) -> tuple[plt.Figure, plt.Axes, collections.PathCollection]:
    """Create a 3D projection scatter plot of the given data.

    Parameters
    ----------
    data : np.ndarray
        The 3D data to plot.
    colors : list[int], optional
        Colors for the data points.

    Returns
    -------
    plt.Figure
        The figure.
    plt.Axes
        The axis of the figure.
    collections.PathCollection
        The scatter plot.

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x, y, z = data.T
    scatter = ax.scatter(x, y, z, c=colors)

    return fig, ax, scatter


def create_3d_animation_scatter_plot(
    data: np.ndarray,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Create a 3D animation scatter plot in PC space."""
    # Get the number of frames (required for the animation)
    # and create an array representing the data point colors, which
    # are to be animated.
    n_frames = data.shape[0]
    colors = [1 for _ in range(n_frames)]

    def change_color_at_index(i) -> list:
        c = colors.copy()
        c[i] = 2
        return c

    # Create the scatter plot with color at data point zero (day 1)
    # changed.
    fig, _, scatter = create_3d_scatter_plot(
        data,
        colors=change_color_at_index(0),
    )
    fig.suptitle("Day 1")

    def update_colors(i):
        """Change the color of the data point at index `i`.

        This data point represents day `i + 1`.

        """
        scatter.set_array(change_color_at_index(i))
        fig.suptitle(f"Day {i + 1}")

    # Create the animation.
    anim = animation.FuncAnimation(
        fig, update_colors, frames=n_frames, interval=500
    )
    return fig, anim


def create_timeseries_scatter_plot(
    x: np.array,
    y: np.array,
    colors: Optional[list[int]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xticks: Optional[list] = None,
    yticks: Optional[list] = None,
) -> tuple[plt.Figure, collections.PathCollection]:
    """Create a timeseries plot."""
    fig = plt.figure()
    scatter = plt.scatter(x, y, c=colors)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if xlabel is not None:
        plt.ylabel(ylabel)

    if xticks is not None:
        plt.xticks(xticks)

    if yticks is not None:
        plt.yticks(yticks)

    return fig, scatter


def create_label_time_series_scatter_plot(
    data: xr.Dataset,
    clusters: Union[cluster.KMeans, hdbscan.HDBSCAN],
    time_coordinate: str = "time",
) -> tuple[plt.Figure, collections.PathCollection]:
    """Create a label time series plot."""
    return create_timeseries_scatter_plot(
        x=data[time_coordinate].values,
        y=clusters.labels_,
        colors=clusters.labels_,
        xlabel="time",
        ylabel="label",
        yticks=list(set(clusters.labels_)),
    )
