import a6.types as types
import numpy as np
import skimage.measure
import xarray as xr


_POOLING_MODES = {
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
    "min": np.min,
}


def apply_pooling(
    data: types.Data, size: float | tuple[float, float], mode: str = "mean"
) -> xr.DataArray:
    """Apply pooling on an image.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        2D data whose size to reduce.
    size : float or tuple[float, float]
        Size of the pooling block.
    mode : str, default="mean"
        Mode to use for the pooling.
        One of `mean`, `median`, `max` or `min`.

    Returns
    -------
    np.ndarray
        Reduced data.

    Notes
    -----
    If the size of the data is not perfectly divisible by the pooling block
    size, the padding value used (`cval`) is according to `mode`. E.g. for
    `mode="mean"`, `cval=np.mean(data)`.

    """
    try:
        func = _POOLING_MODES[mode]
    except KeyError:
        raise ValueError(f"Pooling mode '{mode} not supported")
    return skimage.measure.block_reduce(
        data,
        block_size=size,
        func=func,
        cval=func(data),
    )
