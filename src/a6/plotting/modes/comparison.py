import dataclasses
import datetime
import math
from collections.abc import Iterator

import matplotlib.pyplot as plt
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.datasets.variables as _variables
import a6.features.methods as methods
import a6.plotting.coastlines as _coastlines
import a6.plotting.modes.geopotential as geopotential
import a6.types as types


def plot_fields_for_dates(
    field: xr.DataArray,
    dates: list[datetime],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot given field for a set of dates in the timeseries.

    Parameters
    ----------
    field : xr.DataArray
        The field to plot.
    dates : list[datetime.datetime]
        The dates in the timeseries for which to plot the field.

    """
    figure = _Figure.from_dates_and_field(
        dates=dates,
        field=field,
    )

    vmin = figure.min()
    vmax = figure.max()

    for ax, step in figure.axes_and_fields:
        _coastlines.plot(step, ax=ax, vmin=vmin, vmax=vmax)

    return figure.fig, figure.axs


def plot_contours_for_field_and_dates(
    field: xr.DataArray,
    dates: list[datetime.datetime],
    steps: int | None = 5,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot contours for a given field and dates in the timeseries.

    Parameters
    ----------
    field : xr.DataArray
        The field to plot the contours for.
    dates : list[datetime.datetime]
        The dates in the timeseries for which to plot the contours.
    steps : int, default=5
        The steps to use for the contour heights.

    """
    figure = _Figure.from_dates_and_field(
        dates=dates,
        field=field,
    )

    for ax, step in figure.axes_and_fields:
        geopotential.plot_geopotential_height_contours(
            data=step,
            steps=steps,
            fig=figure.fig,
            ax=ax,
        )

    return figure.fig, figure.axs


def plot_wind_speed_for_dates(
    field: xr.Dataset,
    dates: list[datetime.datetime],
    variables: _variables.Model = _variables.Model(),
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
    steps: int | None = 20,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot contours for a given field and dates in the timeseries.

    Parameters
    ----------
    field : xr.DataArray
        The field to plot the contours for.
    dates : list[datetime.datetime]
        The dates in the timeseries for which to plot the contours.
    variables : a6.datasets.variables.Model, optional
        Weather model variable names.
    coordinates : a6.datasets.coordinates.Coordinates, optional
        Coordinate names.
    steps : int, default=5
        The number of data points to skip for plotting the vectors.

    """
    field = methods.wind.calculate_wind_speed(variables=variables).apply_to(
        field.copy(deep=True)
    )
    figure = _Figure.from_dates_and_field(
        dates=dates,
        field=field,
    )
    vmin = figure.min(field=variables.wind_speed)
    vmax = figure.max(field=variables.wind_speed)
    levels = _get_levels_for_wind_speed(vmax)

    for ax, step in figure.axes_and_fields:
        step["u_norm"] = _normalize_vectors(
            step[variables.u], step[variables.wind_speed]
        )
        step["v_norm"] = _normalize_vectors(
            step[variables.v], step[variables.wind_speed]
        )

        _coastlines.plot_contourf(
            step[variables.wind_speed],
            ax=ax,
            levels=levels,
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )

        sub = _get_subset_for_vector_plot(data=step, steps=steps)
        sub.plot.quiver(
            ax=ax,
            x=coordinates.longitude,
            y=coordinates.latitude,
            u="u_norm",
            v="v_norm",
            scale=steps,
        )

    return figure.fig, figure.axs


def plot_combined(  # noqa: CFQ002
    data: xr.Dataset,
    dates: list[datetime.datetime],
    geopotential_height: str = "z_h",
    variables: _variables.Model = _variables.Model(),
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
    vector_steps: int | None = 20,
    contour_steps: int | None = 5,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot geopotential height contours, temperature and wind speed."""
    figure = _Figure.from_dates_and_field(
        dates=dates,
        field=data,
    )

    vmin = figure.min(field=variables.t)
    vmax = figure.max(field=variables.t)

    for ax, step in figure.axes_and_fields:
        step[variables.t].plot(
            ax=ax,
            cmap="RdBu",
            vmin=vmin,
            vmax=vmax,
        )

        geopotential.plot_geopotential_height_contours(
            data=step[geopotential_height],
            steps=contour_steps,
            fig=figure.fig,
            ax=ax,
            cmap="black",
        )

        sub = _get_subset_for_vector_plot(data=step, steps=vector_steps)
        sub.plot.quiver(
            ax=ax,
            x=coordinates.longitude,
            y=coordinates.latitude,
            u=variables.u,
            v=variables.v,
            scale=250,
        )

    return figure.fig, figure.axs


def _normalize_vectors(
    data: types.DataND, normalization: types.DataND
) -> types.DataND:
    return data / normalization


def _get_levels_for_wind_speed(vmax: float) -> int:
    return round(vmax / 2)


def _get_subset_for_vector_plot(data: types.DataND, steps: int) -> types.DataND:
    skip = slice(None, None, steps)
    return data.sel(longitude=skip, latitude=skip)


@dataclasses.dataclass
class _Figure:
    fig: plt.Figure
    axs: plt.Axes
    fields: types.XarrayData

    @classmethod
    def from_dates_and_field(
        cls,
        dates: list[datetime.datetime],
        field: types.XarrayData,
    ) -> "_Figure":
        n_rows = math.ceil(len(dates) / 3)
        n_cols = 3

        height = n_rows * 6
        width = n_cols * 8

        fig, axs = plt.subplots(
            figsize=(width, height),
            nrows=n_rows,
            ncols=n_cols,
            subplot_kw=_coastlines.create_projection(),
        )

        fields = field.sel(time=dates)

        return cls(
            fig=fig,
            axs=axs,
            fields=fields,
        )

    @property
    def axes(self) -> Iterator[plt.Axes]:
        yield from self.axs.flatten()

    @property
    def axes_and_fields(self) -> Iterator[tuple[plt.Axes, types.XarrayData]]:
        if isinstance(self.fields, xr.DataArray):
            fields = self.fields
        elif isinstance(self.fields, xr.Dataset):
            fields = (
                self.fields.sel(time=step) for step in self.fields["time"]
            )
        yield from zip(self.axes, fields)

    def max(self, field: str | None = None) -> float:
        if isinstance(self.fields, xr.DataArray):
            return float(self.fields.max().values)
        elif isinstance(self.fields, xr.Dataset):
            return float(self.fields[field].max().values)

    def min(self, field: str | None = None) -> float:
        if isinstance(self.fields, xr.DataArray):
            return float(self.fields.min().values)
        elif isinstance(self.fields, xr.Dataset):
            return float(self.fields[field].min().values)
