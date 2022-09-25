import pathlib

import xarray as xr

# Create a sorted list of the files in the given path.
path = pathlib.Path("/p/project/training2223/a6/data/ml")
paths = sorted(path.glob("ml_2020010*.nc"))


def slice_first_twelve_time_steps(ds: xr.Dataset) -> xr.Dataset:
    return ds.isel(time=slice(None, 12))


# Open all files one after another, apply a preprocessing function
# and append each file to the dataset.
ds = xr.open_mfdataset(
    paths,
    preprocess=slice_first_twelve_time_steps,
)
