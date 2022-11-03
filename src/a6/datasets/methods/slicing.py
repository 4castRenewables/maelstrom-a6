import a6.utils as utils
import xarray as xr


@utils.make_functional
def slice_dataset(
    dataset: xr.Dataset,
    dimension: str,
    slice_until: int,
    slice_from: int | None = None,
) -> xr.Dataset:
    """Drop data before/after given indexes of given dimension.

    Parameters
    ----------
    dataset : xr.Dataset
    dimension : str
        Name of the dimension along which to slice.
    slice_until : int
        Index until which to slice the data along given dimension.
    slice_from : int, optional
        Index from which to slice the data along given dimension.
        If `None`, slice from 0-th index.

    Returns
    -------
    xr.Dataset
        Contains only the data within the given indexes.

    """
    return dataset.isel({dimension: slice(slice_from, slice_until)})
