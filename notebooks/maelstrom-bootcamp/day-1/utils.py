from typing import Optional

import matplotlib.pyplot as plt
import xarray as xr


def create_twin_y_axis_plot(
    data: xr.Dataset,
    left: str,
    right: str,
    left_ylim: Optional[tuple[float, float]] = None,
    right_ylim: Optional[tuple[float, float]] = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create a twin axis plot with two Y-axes.

    Parameters
    ----------
    data : xr.Dataset
        Dataset from which to plot two given quantities.
    left : str
        Name of the quantity to plot on the left y-axis.
    right : str
        Name of the quantity to plot on the right y-axis.
    left_ylim : tuple[float, float], default=(0, 1.1 * max(left))
        Limits for the left y-axis.
    right_ylim : tuple[float, float], default=(0, 1.1 * max(right))
        Limits for the right y-axis.

    Returns
    -------
    plt.Figure
        The figure instance.
    tuple[plt.Axes, plt.Axes]
        The left and right axis, respectively.

    """
    left_da = data[left]
    right_da = data[right]

    # Create the figure and axis.
    fig, ax1 = plt.subplots()

    # Plot the left y-axis.
    left_da.plot(axes=ax1)

    if left_ylim is None:
        # Adapt the y limit of the left y-axis.
        ax1.set_ylim(*_create_limit(left_da))

    # Create a second axis and plot the wind speed on it.
    ax2 = ax1.twinx()

    # Plot the right y-axis.
    right_da.plot(axes=ax2, color="orange")

    if right_ylim is None:
        # Adapt the y limit of the right y-axis.
        ax2.set_ylim(*_create_limit(right_da))

    return fig, (ax1, ax2)


def _create_limit(data: xr.DataArray) -> tuple[float, float]:
    return 0, 1.1 * data.max()
