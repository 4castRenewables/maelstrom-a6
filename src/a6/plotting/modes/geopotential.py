import typing as t

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_geopotential_height_contours(
    data: xr.DataArray,
    temperature: t.Optional[xr.DataArray] = None,
    steps: int = 5,
    fig: t.Optional[plt.Figure] = None,
    ax: t.Optional[plt.Axes] = None,
    cmap: t.Optional[str] = None,
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

    def round_to_decade(value: xr.DataArray) -> int:
        return int(np.round(value.values, -1))

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    levels = range(
        round_to_decade(data.min()),
        round_to_decade(data.max()),
        steps,
    )

    if temperature is not None:
        temperature.plot(ax=ax, cmap="Greys")

    # Geopotential is given in decameters in ECMWF IFS HRES.
    contours = data.plot.contour(
        levels=levels, kwargs=dict(inline=True), ax=ax, cmap=cmap
    )
    ax.clabel(contours)

    return fig, ax
