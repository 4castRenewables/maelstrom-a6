import logging
import pathlib
import time

import xarray as xr

import a6.utils as utils

logger = logging.getLogger(__name__)


@utils.log_consumption
@utils.make_functional
def to_netcdf(ds: xr.Dataset, *, path: pathlib.Path) -> xr.Dataset:
    """Save dataset to netcdf file."""
    logger.info("Saving dataset %s to disk at %s", ds, path)
    start = time.time()
    ds.to_netcdf(path)
    logger.info("Saving finished in %.2f seconds", time.time() - start)
    return ds
