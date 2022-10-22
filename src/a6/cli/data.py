import logging
import pathlib

import xarray as xr

logger = logging.getLogger(__name__)


def read(path: pathlib.Path, **kwargs) -> xr.Dataset:
    """Read a given data path and convert to `xarray`.

    Parameters
    ----------
    path : pathlib.Path
        Path to the data.
    kwargs : dict
        Selection criteria.

    Returns
    -------
    xr.Dataset

    """
    logger.debug("Reading data from %s with select criteria %s", path, kwargs)

    ds = xr.open_dataset(path)
    if kwargs:
        select = {k: v for k, v in kwargs.items() if v is not None}
        return ds.sel(select)
    return ds
