import logging
import pathlib
from collections.abc import Callable
from typing import TypeVar

import xarray as xr

import a6.datasets.methods as methods
import a6.types as types
import a6.utils as utils


logger = logging.getLogger(__name__)

Processing = TypeVar(
    "Processing", utils.Functional, Callable[[xr.Dataset], xr.Dataset]
)


class Dataset:
    """Represents a dataset."""

    _engine = "netcdf4"
    _concat_dim = "time"

    def __init__(  # noqa: CFQ002
        self,
        path: pathlib.Path,
        pattern: str | None = "*.nc",
        slice_time_dimension: bool = True,
        slice_time_time_dimension_after: int | None = None,
        preprocessing: Processing | None = None,
        postprocessing: Processing | None = None,
        parallel_loading: bool = True,
    ):
        """Initialize without opening the files.

        Parameters
        ----------
        path : pathlib.Path
            Paths to the data file or folder.
        pattern : str, default="*.nc"
            Pattern for the data files to read.
            If path is a file, the pattern will be ignored.
        slice_time_dimension : bool, default=True
            Whether the files are temporarily overlapping and should be sliced.
            The ECMWF models are usually run e.g. at 00am, 06am, 12pm, 18pm
            for 48 hours. As a consequence, the data of new models overlap
            with data from older models by some hours (6 in this case).
        slice_time_time_dimension_after : int, optional
            The number of time steps after which to slice each data file.
        preprocessing : Callable, optional
            Pre-processing to apply to each data file before appending i
        postprocessing : Callable, optional
            Post-processing to apply to the entire dataset after reading
            individual files.
        parallel_loading : bool, default True
            Whether to load the data files parallely.

        """
        if path.is_file():
            self.paths = [path]
        else:
            self.paths = utils.list_files(path=path, pattern=pattern)

            if not self.paths:
                logger.warning(
                    "No files found %s in with pattern %s",
                    path,
                    pattern,
                )

        self._slice_time_dimension = slice_time_dimension
        self._slice_time_dimension_after = slice_time_time_dimension_after
        self._parallel = parallel_loading
        self._preprocessing = preprocessing
        self._postprocessing = postprocessing

        self._data: xr.Dataset | None = None
        self._dropped_variables = []
        self._selected_levels = []

    @utils.log_consumption
    def to_xarray(
        self,
        levels: types.Levels = None,
        drop_variables: list[str] | None = None,
    ) -> xr.Dataset:
        """Return the dataset as an `xr.Dataset`.

        Parameters
        ----------
        levels : int or list[int], optional
            Level(s) to selec
        drop_variables : list[str], optional
            List of variables to drop from the datase

        """
        logger.debug(
            "Selecting level %s and dropping variables %s",
            levels,
            drop_variables,
        )
        if self._was_already_converted(
            levels=levels, drop_variables=drop_variables
        ):
            logger.debug(
                "Data was already converted for levels %s and dropped "
                "variables %s",
                levels,
                drop_variables,
            )
            return self._data
        return self._to_xarray(levels=levels, drop_variables=drop_variables)

    def _was_already_converted(
        self, levels: types.Levels, drop_variables: list[str] | None
    ) -> bool:
        return (
            self._data is not None
            and drop_variables == self._dropped_variables
            and levels == self._selected_levels
        )

    def _to_xarray(
        self, levels: types.Levels, drop_variables: list[str] | None
    ) -> xr.Dataset:
        """Merge a set of files into a single datase"""
        if len(self.paths) == 1:
            ds = self._open_single_dataset(drop_variables=drop_variables)
        else:
            ds = self._open_multiple_temporally_monotonous_datasets(
                drop_variables=drop_variables
            )

        if self._postprocessing is not None:
            return self._postprocessing(ds)

        if levels is not None:
            logger.debug("Selecting level %s", levels)
            return ds.sel(level=levels)
        return ds

    def _open_single_dataset(
        self, drop_variables: list[str] | None
    ) -> xr.Dataset:
        [path] = self.paths
        dataset = xr.open_dataset(
            path,
            engine=self._engine,
            drop_variables=drop_variables,
        )
        return self._preprocess_dataset(dataset)

    def _open_multiple_temporally_monotonous_datasets(
        self, drop_variables: list[str] | None
    ):
        return xr.open_mfdataset(
            self.paths,
            engine=self._engine,
            concat_dim=self._concat_dim,
            combine="nested",
            coords="minimal",
            data_vars="minimal",
            preprocess=self._preprocess_dataset,
            compat="override",
            parallel=self._parallel,
            drop_variables=drop_variables,
        )

    def _preprocess_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        if self._slice_time_dimension:
            dataset = self._slice_first_twelve_hours(dataset)
        if self._preprocessing is not None:
            return self._preprocessing(dataset)
        return dataset

    def _slice_first_twelve_hours(self, dataset: xr.Dataset) -> xr.Dataset:
        """Cut an hourly dataset after the first 12 hours.

        This is necessary to overwrite older model runs with newer ones.
        Models are calculated e.g. at 00:00 and 12:00 for 48 hours each. Hence,
        when using two model runs, one always wants the newer values of the
        recent run to overwrite the older ones.

        """
        return methods.slicing.slice_dataset(
            dataset,
            dimension=self._concat_dim,
            slice_until=self._slice_time_dimension_after,
            non_functional=True,
        )
