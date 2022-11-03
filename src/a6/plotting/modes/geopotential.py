import a6.plotting.coastlines as _coastlines
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_geopotential_height_contours(
    data: xr.DataArray,
    steps: int = 5,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    cmap: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the geopotential height contours.

    Parameters
    ----------
    data : xr.DataArray
        Geopotential heights.
    steps : int, default=5
        Steps in hPa for the contour levels.

    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(subplot_kw=_coastlines.create_projection())

    min = int(np.round(data.min().values, -1))
    max = int(np.round(data.max().values, -1))
    levels = range(min, max, steps)

    # Geopotential is given in decameters in ECMWF IFS HRES.
    ax = _coastlines.plot_contour(
        data, ax=ax, levels=levels, kwargs=dict(inline=True), cmap=cmap
    )

    return fig, ax
