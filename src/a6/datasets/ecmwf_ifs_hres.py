import logging
import pathlib
import typing as t

import a6.utils
import xarray as xr

Path = t.Union[str, pathlib.Path]
Levels = t.Optional[t.Union[int, list[int]]]

logger = logging.getLogger(__name__)


class EcmwfIfsHres:
    """Represents the IFS HRES data from ECMWF."""

    _engine = "netcdf4"
    _concat_dim = "time"

    def __init__(
        self,
        paths: t.Union[Path, list[Path]],
        overlapping: bool = True,
        preprocessing: t.Optional[t.Callable[[xr.Dataset], xr.Dataset]] = None,
        postprocessing: t.Optional[t.Callable[[xr.Dataset], xr.Dataset]] = None,
        parallel_loading: bool = True,
    ):
        """Initialize without opening the files.

        Parameters
        ----------
        paths : str | pathlib.Path or list[str | pathlib.Path]
            Paths to the data files.
        overlapping : bool, default=True
            Whether the files are temporarily overlapping.
            The ECMWF models are usually run at 12am and 12 pm for 48 hours.
            As a consequence, the data of new models overlap with data from
            older models by 12 hours.
        preprocessing : Callable, optional
            Pre-processing to apply to each data file before appending it.
        postprocessing : Callable, optional
            Post-processing to apply to the entire dataset after reading
            individual files.
        parallel_loading : bool, default True
            Whether to load the data files parallely.

        """
        if not paths:
            raise ValueError("No source paths given")

        self.paths = paths
        self._overlapping = overlapping
        self._parallel = parallel_loading
        self._preprocessing = preprocessing
        self._postprocessing = postprocessing

        self._data: t.Optional[xr.Dataset] = None
        self._dropped_variables = []
        self._selected_levels = []

    @a6.utils.log_consumption
    def as_xarray(
        self,
        levels: Levels = None,
        drop_variables: t.Optional[list[str]] = None,
    ) -> xr.Dataset:
        """Return the dataset as an `xr.Dataset`.

        Parameters
        ----------
        levels : int or list[int], optional
            Level(s) to select.
        drop_variables : list[str], optional
            List of variables to drop from the dataset.

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
        return self._as_xarray(levels=levels, drop_variables=drop_variables)

    def _was_already_converted(
        self, levels: Levels, drop_variables: t.Optional[list[str]]
    ) -> bool:
        return (
            self._data is not None
            and drop_variables == self._dropped_variables
            and levels == self._selected_levels
        )

    def _as_xarray(
        self, levels: Levels, drop_variables: t.Optional[list[str]]
    ) -> xr.Dataset:
        """Merge a set of files into a single dataset."""
        if _single_path_given(self.paths):
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
        self, drop_variables: t.Optional[list[str]]
    ) -> xr.Dataset:
        if isinstance(self.paths, list):
            [path] = self.paths
        else:
            path = self.paths

        dataset = xr.open_dataset(
            path,
            engine=self._engine,
            drop_variables=drop_variables,
        )
        if self._preprocessing is not None:
            return self._preprocessing(dataset)
        return dataset

    def _open_multiple_temporally_monotonous_datasets(
        self, drop_variables: t.Optional[list[str]]
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
        if self._overlapping:
            dataset = self._slice_first_twelve_hours(dataset)
        if self._preprocessing is not None:
            return self._preprocessing(dataset)
        return dataset

    def _slice_first_twelve_hours(self, dataset: xr.Dataset) -> xr.Dataset:
        """Cut an hourly dataset after the first 12 hours.

        This is necessary to overwrite older model runs with newer ones.
        Models are calculated at 00:00 and 12:00 for 48 hours each. Hence,
        when using two model runs, one always wants the newer values of the
        recent run to overwrite the older ones.

        """
        return a6.utils.slice_dataset(
            dataset,
            dimension=self._concat_dim,
            slice_until=12,
        )


def _single_path_given(paths: t.Union[Path, list[Path]]) -> bool:
    return isinstance(paths, (str, pathlib.Path)) or len(paths) == 1
