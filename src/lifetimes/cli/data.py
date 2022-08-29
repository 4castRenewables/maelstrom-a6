import typing as t

import lifetimes.datasets
import xarray as xr


def read_ecmwf_ifs_hres_data(path: str, level: t.Optional[int]) -> xr.Dataset:
    """Read a given data path and convert to `xarray`.

    Parameters
    ----------
    path : str
        Path to the data.
    level : int, optional
        Level for which to perform the procedure.

    Returns
    -------
    xr.Dataset

    """
    ds = lifetimes.datasets.EcmwfIfsHres(
        paths=[path],
        overlapping=False,
    )
    converted = ds.as_xarray()

    if level is not None:
        return converted.sel(level=level)
    return converted
