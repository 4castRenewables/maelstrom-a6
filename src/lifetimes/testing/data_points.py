import lifetimes.testing.data_factories as data_factories
import lifetimes.testing.grids as grids
import lifetimes.testing.types as types
import pandas as pd
import xarray as xr


class DataPoints:
    """A set of data points on an arbitrary grid with a certain lifetime.

    Extends an `xarray.Dataset`, which represents a timeseries with grid data,
    with values (on that same grid) at certain time steps.

    Values are defined via `GridDataFactory`, which enable adding values at
    specific spatial coordinates to a grid. The given timeseries grid data are
    then extended by adding the values at each time step starting at `start`
    until `end` at intervals `frequency` (given as `string`).

    """

    def __init__(
        self,
        data_factory: data_factories.GridDataFactory,
        start: types.Timestamp,
        end: types.Timestamp,
        frequency: str,
    ):
        self.factory = data_factory
        self.dates = pd.date_range(start, end, freq=frequency)

    def add_grid_to_timeseries(
        self,
        timeseries: xr.DataArray,
        time_coordinate: str,
        grid: grids.Grid,
    ) -> xr.DataArray:
        """Adds data points (values defined on a grid) to a timeseries at
        overlapping times of `DataPoints` and `timeseries`.

        Parameters
        ----------
        timeseries : xr.DataArray
            Timeseries data to which to add the data.
        time_coordinate : str
            Name of the time coordinate of the timeseries.
        grid : lifetimes.testing.grids.Grid
            Grid on which to create the data.

        Returns
        -------
        xr.DataArray
            Timeseries with the data added to the time steps within
            the defined date range.

        """
        timesteps_within_range = self._determine_timesteps_within_range(
            timeseries=timeseries, time_coordinate=time_coordinate
        )
        timeseries = self._add_data_to_timeseries(
            timeseries=timeseries,
            time_coordinate=time_coordinate,
            timesteps=timesteps_within_range,
            grid=grid,
        )
        return timeseries

    def _determine_timesteps_within_range(
        self, timeseries: xr.DataArray, time_coordinate: str
    ) -> xr.DataArray:
        start = self.dates[0]
        end = self.dates[-1]
        condition = (start <= timeseries[time_coordinate]) & (
            timeseries[time_coordinate] <= end
        )
        timesteps_within_range = timeseries[time_coordinate][condition]
        return timesteps_within_range

    def _add_data_to_timeseries(
        self,
        timeseries: xr.DataArray,
        time_coordinate: str,
        timesteps: xr.DataArray,
        grid: grids.Grid,
    ) -> xr.DataArray:
        data = self.factory.create(grid)
        timeseries.loc[{time_coordinate: timesteps}] += data
        return timeseries
