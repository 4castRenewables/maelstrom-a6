import a6.types as types
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def create_projection() -> dict:
    """Create projection for coastlines."""
    return {
        "projection": ccrs.PlateCarree(),
    }


def plot(
    data: types.XarrayData, ax: plt.Axes | None = None, **kwargs
) -> plt.Axes:
    """Plot the given data with coastlines."""
    ax = _create_axis(ax)
    data.plot(**kwargs, ax=ax, transform=ccrs.PlateCarree())
    return _draw_lines(ax)


def plot_contour(
    data: types.XarrayData, ax: plt.Axes | None = None, **kwargs
) -> plt.Axes:
    """Plot the given data with coastlines."""
    ax = _create_axis(ax)
    _axis = data.plot.contour(**kwargs, ax=ax, transform=ccrs.PlateCarree())
    _axis.clabel()
    return _draw_lines(ax)


def plot_contourf(
    data: types.XarrayData, ax: plt.Axes | None = None, **kwargs
) -> plt.Axes:
    """Plot the given data with coastlines."""
    ax = _create_axis(ax)
    _axis = data.plot.contourf(**kwargs, ax=ax, transform=ccrs.PlateCarree())
    _axis.clabel()
    return _draw_lines(ax)


def _create_axis(ax: plt.Axes | None) -> plt.Axes:
    if ax is None:
        return plt.subplot(**create_projection())
    return ax


def _draw_lines(ax: plt.Axes) -> plt.Axes:
    ax.coastlines()
    ax.gridlines(draw_labels=["left", "bottom"])
    return ax
