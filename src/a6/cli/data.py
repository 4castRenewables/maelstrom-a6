import logging
import pathlib

import xarray as xr

import a6.datasets as datasets

logger = logging.getLogger(__name__)


def read(
    path: pathlib.Path, pattern: str, slice_files: bool, level: int | None
) -> xr.Dataset:
    """Read a given data path and convert to `xarray`.

    Parameters
    ----------
    path : pathlib.Path
        Path to the data.
    pattern : str
        Pattern for the data files to read.
    slice_files : bool
        Whether to slice each data file.
    level : int, optional
        Level to select.

    Returns
    -------
    xr.Dataset

    """
    ds = datasets.EcmwfIfsHres(
        path=path, pattern=pattern, slice_time_dimension=slice_files
    )
    return ds.to_xarray(levels=level)
