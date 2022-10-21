import logging
import pathlib
import typing as t

import a6.datasets
import xarray as xr

logger = logging.getLogger(__name__)


def read_ecmwf_ifs_hres_data(
    path: pathlib.Path, level: t.Optional[int]
) -> xr.Dataset:
    """Read a given data path and convert to `xarray`.

    Parameters
    ----------
    path : pathlib.Path
        Path to the data.
    level : int, optional
        Level for which to perform the procedure.

    Returns
    -------
    xr.Dataset

    """
    logger.debug("Reading data from %s", path)
    ds = a6.datasets.EcmwfIfsHres(
        paths=[path],
        overlapping=False,
    )

    logger.debug("Converting to xarray")
    converted = ds.as_xarray()

    if level is not None:
        logger.debug("Selecting level %s", level)
        return converted.sel(level=level)
    return converted
