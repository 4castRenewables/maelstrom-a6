import typing as t

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_geopotential_height_contours(
    data: xr.DataArray,
    temperature: t.Optional[xr.DataArray] = None,
    steps: int = 5,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the geopotential height contours.

    Parameters
    ----------
    data : xr.DataArray
        Geopotential heights.
    temperature : str, optional
        Temperature field for the same time step.
        Temperature will be plotted to identify land mass.
    steps : int, default=5
        Steps in hPa for the contour levels.

    """
    fig, ax = plt.subplots()

    min = int(np.round(data.min().values, -1))
    max = int(np.round(data.max().values, -1))
    levels = range(min, max, steps)

    if temperature is not None:
        temperature.plot(ax=ax, cmap="Greys")

    # Geopotential is given in decameters in ECMWF IFS HRES.
    contours = data.plot.contour(levels=levels, kwargs=dict(inline=True), ax=ax)
    ax.clabel(contours)

    return fig, ax
