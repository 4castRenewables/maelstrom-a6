import typing as t

import lifetimes.modes.methods.pca as _pca
import matplotlib.pyplot as plt
import numpy as np


def plot_scree_test(
    pca: _pca.PCA,
    variance_ratio: t.Optional[float] = None,
    display: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create a scree test plot for a PCA.

    Parameters
    ----------
    pca : lifetimes.modes.methods.pca.PCA
    variance_ratio : float, optional
        Variance ratio excess at which to draw a vertical line.
        Indicates the number of components needed to achieve given variance.
    display: bool, default=True
        Whether display the plot, i.e. call `plt.show()`.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    """
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("number of components")

    x = list(range(1, pca.n_components + 1))

    # Plot cumulative variance on first axis
    color_ax1 = "tab:red"
    ax1.set_ylabel("cumulative explained variance", color=color_ax1)
    ax1.scatter(x, pca.cumulative_variance_ratios, color=color_ax1)

    # Create right axis.
    ax2 = ax1.twinx()

    # Plot the explained variance ratios.
    color_ax2 = "tab:blue"
    ax2.set_ylabel("explained variance ratio", color=color_ax2)
    ax2.scatter(x, pca.variance_ratios, color=color_ax2)

    for ax, color in [(ax1, color_ax1), (ax2, color_ax2)]:
        # Set log scale.
        ax.set(xscale="log", yscale="log")
        # Set left xlim such that the first tick disappears.
        ax.set_xlim(0.91, None)
        # Color the ticks.
        ax.tick_params(axis="y", colors=color, which="both")

    # Plot vertical lince indicating variance excess.
    if variance_ratio is not None:
        n_components = pca.number_of_components_sufficient_for_variance_ratio(
            variance_ratio
        )
        # Dashed line indicating the threshold.
        ax2.axvline(
            n_components,
            ymin=0,
            ymax=2 * np.max(pca.variance_ratios.values),
            linestyle="dashed",
            color="grey",
        )
        ax2.text(
            n_components + 0.25,
            np.min(pca.variance_ratios),
            f"$n_{{comp}} = {n_components}$",
            rotation=90,
            color="grey",
        )

    fig.tight_layout()

    if display:
        plt.show()

    return fig, (ax1, ax2)


def plot_first_three_components_timeseries(
    pca: _pca.PCA, colors: t.Optional[list] = None, display: bool = True
) -> tuple[plt.Figure, plt.Axes]:
    """Return a figure with the first three PCs plotted."""

    pc_timeseries = pca.transform(n_components=3)
    x, y, z = pc_timeseries.T

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x, y, z, c=colors)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    if display:
        plt.show()

    return fig, ax
