import xarray as xr


def get_number_of_input_channels(ds: xr.Dataset | xr.DataArray) -> int:
    """Get the number of input channels required for ``torch.nn.Conv2d``.

    This is given by the number of phyiscal quantities contained in the
    data.

    """
    if isinstance(ds, xr.DataArray):
        # ``xr.DataArray`` always only contains a single variable.
        return 1
    return len(ds.data_vars)
