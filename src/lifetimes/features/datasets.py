import abc
import pathlib
import typing as t

import lifetimes.utils
import xarray as xr

Path = t.Union[str, pathlib.Path]


class Dataset(abc.ABC):
    """A dataset."""

    # Variables dropped when the dataset was previously
    # converted to an `xr.Dataset`.
    _dropped_variables: list[str]

    def __init__(self):
        self._data: t.Optional[xr.Dataset] = None
        self._dropped_variables = []

    @lifetimes.utils.log_runtime
    def as_xarray(
        self, drop_variables: t.Optional[list[str]] = None
    ) -> xr.Dataset:
        """Return the dataset as an `xr.Dataset`.

        Parameters
        ----------
        drop_variables : list[str], optional
            List of variables to drop from the dataset.

        """
        if self._data is not None and drop_variables == self._dropped_variables:
            return self._data
        return self._as_xarray(drop_variables=drop_variables)

    @abc.abstractmethod
    def _as_xarray(self, drop_variables: t.Optional[list[str]]) -> xr.Dataset:
        ...


class FileDataset(Dataset):
    """A dataset read from files on local storage."""

    def __init__(self, paths: list[Path]):
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
        paths: list[Path],
        overlapping: bool,
        preprocessing: t.Optional[t.Callable[[xr.Dataset], xr.Dataset]] = None,
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
        self._overlapping = overlapping
        self._parallel = parallel_loading
        self._preprocessing = preprocessing

    def _as_xarray(self, drop_variables: t.Optional[list[str]]) -> xr.Dataset:
        """Merge a set of files into a single dataset."""
        if len(self.paths) == 1:
            return self._open_single_dataset(drop_variables=drop_variables)
        return self._open_multiple_temporally_monotonous_datasets(
            drop_variables=drop_variables
        )

    def _open_single_dataset(
        self, drop_variables: t.Optional[list[str]]
    ) -> xr.Dataset:
        dataset = xr.open_dataset(
            *self.paths,
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
        return lifetimes.utils.slice_dataset(
            dataset,
            dimension=self._concat_dim,
            slice_until=12,
        )
