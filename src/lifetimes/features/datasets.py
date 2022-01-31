import abc
import pathlib
from typing import Optional
from typing import Union

import xarray as xr


class Dataset(abc.ABC):
    """A dataset."""

    def __init__(self):
        self._data: Optional[xr.Dataset] = None

    def as_xarray(self) -> xr.Dataset:
        """Return the dataset as an xarray Dataset."""
        if self._data is None:
            self._data = self._as_xarray()
        return self._data

    @abc.abstractmethod
    def _as_xarray(self) -> xr.Dataset:
        ...


class FileDataset(Dataset):
    """A dataset read from files on local storage."""

    def __init__(self, paths: list[Union[str, pathlib.Path]]):
        if not paths:
            raise ValueError("No source paths given")
        super().__init__()
        self.paths = paths


class EcmwfIfsHresDataset(FileDataset):
    """Represents the IFS HRES data from ECMWF."""

    _engine = "netcdf4"
    _concat_dim = "time"

    def __init__(
        self,
        paths: list[Union[str, pathlib.Path]],
        overlapping: bool,
        parallel_loading: bool = True,
    ):
        """Initialize without opening the files.

        Parameters
        ----------
        paths : list[str | pathlib.Path]
            Paths to the data files.
        overlapping : bool
            Whether the files are temporarily overlapping.
            The ECMWF models are usually run at 12am and 12 pm for 48 hours.
            As a consequence, the data of new models overlap with data from
            older models by 12 hours.
        parallel_loading : bool, default True
            Whether to load the data files parallely.

        """
        super().__init__(paths)
        self.overlapping = overlapping
        self.parallel = parallel_loading

    def _as_xarray(self) -> xr.Dataset:
        """Merge a set of files into a single dataset."""
        if len(self.paths) == 1:
            return self._open_single_dataset(*self.paths)
        return self._open_multiple_temporally_monotonous_datasets()

    def _open_single_dataset(
        self, path: Union[str, pathlib.Path]
    ) -> xr.Dataset:
        return xr.open_dataset(
            path,
            engine=self._engine,
        )

    def _open_multiple_temporally_monotonous_datasets(self):
        preprocessing = (
            self._slice_first_twelve_hours if self.overlapping else None
        )
        return xr.open_mfdataset(
            self.paths,
            engine=self._engine,
            concat_dim=self._concat_dim,
            combine="nested",
            coords="minimal",
            data_vars="minimal",
            preprocess=preprocessing,
            compat="override",
            parallel=self.parallel,
        )

    def _slice_first_twelve_hours(self, dataset: xr.Dataset) -> xr.Dataset:
        """Cut an hourly dataset after the first 12 hours.

        This is necessary to overwrite older model runs with newer ones.
        Models are calculated at 00:00 and 12:00 for 48 hours each. Hence,
        when using two model runs, one always wants the newer values of the
        recent run to overwrite the older ones.

        """
        return dataset.isel({self._concat_dim: slice(None, 12)})
