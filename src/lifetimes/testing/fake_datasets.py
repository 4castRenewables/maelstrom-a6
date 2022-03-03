import datetime
import functools
import typing as t

import pandas as pd
import xarray as xr
from lifetimes.features import datasets

from . import data_points
from . import grids
from . import types


class FakeDataset(datasets.FileDataset):
    """A fake timeseries dataset for grid data."""

    grid: grids.Grid
    time_coordinate: str

    def __init__(
        self,
        grid: grids.Grid,
        start: t.Union[str, datetime.datetime],
        end: t.Union[str, datetime.datetime],
        frequency: str,
        data: t.Optional[list[data_points.DataPoints]],
    ):
        """Create the dataset.

        By default, a dataset of duration of 50 days is created.

        """
        super().__init__(["test_path"])
        self.grid = grid
        self.dates = pd.date_range(start, end, freq=frequency)
        self.shape = (self.dates.size, *self.grid.shape)
        self.data = data

    @property
    def _dimensions(self) -> list[str]:
        return list(self._coordinates)

    @property
    def _coordinates(self) -> dict:
        return {
            self.time_coordinate: ([self.time_coordinate], self.dates),
            **self.grid.xarray_coords_dict,
        }

    @functools.lru_cache
    def _as_xarray(self, drop_variables: t.Optional[list[str]]) -> xr.Dataset:
        da = xr.DataArray(
            data=0.0,
            dims=self._dimensions,
            coords=self._coordinates,
        )

        da = self._add_data_to_timeseries(da)

        ds = xr.Dataset(
            data_vars={
                "ellipse": da,
            },
            coords=self._coordinates,
        )
        if drop_variables is not None:
            return ds.drop_vars(drop_variables)
        return ds

    def _add_data_to_timeseries(self, timeseries: xr.DataArray):
        if self.data is not None:
            for data in self.data:
                timeseries = data.add_to_grid_timeseries(
                    timeseries=timeseries,
                    grid=self.grid,
                    time_coordinate=self.time_coordinate,
                )
        return timeseries


class FakeEcmwfIfsHresDataset(FakeDataset):
    """Fake dataset for data conform to ECMWF IFS HRES data."""

    grid: grids.Grid
    time_coordinate = "time"

    def __init__(
        self,
        grid: t.Optional[grids.Grid],
        start: types.Timestamp,
        end: types.Timestamp,
        frequency: str,
        data: t.Optional[list[data_points.DataPoints]] = None,
    ):
        """Create IFS HRES dataset.

        By default, a dataset of duration of 50 days is created that
        contains 3 ellipses located on the grid, where one of which is rotated.

        """
        if grid is None:
            grid = grids.EcmwfIfsHresGrid()
        super().__init__(
            grid=grid,
            start=start,
            end=end,
            frequency=frequency,
            data=data,
        )
