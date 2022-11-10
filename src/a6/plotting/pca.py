import matplotlib.pyplot as plt
import numpy as np

import a6.modes.methods.pca as _pca


def plot_scree_test(
    pca: _pca.PCA,
    variance_ratio: float | None = None,
    display: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create a scree test plot for a PCA.

    Parameters
    ----------
    pca : a6.modes.methods.pca.PCA
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
    ax1_color = "tab:red"
    ax1.set_ylabel("cumulative explained variance", color=ax1_color)
    ax1.scatter(x, pca.cumulative_variance_ratio, color=ax1_color)

    # Create right axis.
    ax2 = ax1.twinx()

    # Plot the explained variance ratios.
    ax2_color = "tab:blue"
    ax2.set_ylabel("explained variance ratio", color=ax2_color)
    ax2.scatter(x, pca.explained_variance_ratio, color=ax2_color)

    for ax, color in [(ax1, ax1_color), (ax2, ax2_color)]:
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
            ymax=2 * np.max(pca.explained_variance_ratio.values),
            linestyle="dashed",
            color="grey",
        )
        ax2.text(
            1.03 * n_components,
            np.min(pca.explained_variance_ratio),
            f"$n_{{comp}} = {n_components}$",
            rotation=90,
            color="grey",
        )

    fig.tight_layout()

    if display:
        plt.show()

    return fig, (ax1, ax2)


def plot_first_three_components_timeseries(
    pca: _pca.PCA, colors: list | None = None, display: bool = True
) -> tuple[plt.Figure, plt.Axes]:
    """Return a figure with the first three PCs plotted."""

    pc_timeseries = pca.transform(n_components=3)
    x, y, z = pc_timeseries.T

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x, y, z, c=colors, s=4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    if display:
        plt.show()

    return fig, ax
