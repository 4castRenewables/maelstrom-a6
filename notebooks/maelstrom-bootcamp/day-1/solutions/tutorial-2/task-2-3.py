from datetime import datetime

import xarray as xr


ds = xr.open_dataset("/p/project1/training2223/a6/data/ml/ml_20200101_00.nc")

# A
# The coordinates and dimensions of the dataset can be separately accessed.
dimensions = ds.dims
coordinates = ds.coords
print(f"Dataset dimensions:\n\n{dimensions}\n\n")
print(f"Dataset coordinates:\n\n{coordinates}\n\n")

# B
# The dataset allows storing metadata in its attributes.
attributes = ds.attrs
print(f"Dataset metadata (attributes):\n\n{attributes}\n\n")

# C
# The data variables can directly be accessed.
variables = ds.data_vars
print(f"Dataset variables :\n\n{ds.data_vars}\n\n")

# D
# The above variables can also directly be accessed using indexing
# and the variable name. This returns a different object: an `xarray.DataArray`.
# These are quite similar to `numpy.array` or `pandas.DataFrame`.
variable = ds["t"]
print(f"Access a variable of the dataset:\n\n{variable}\n\n")

# The metadata of each variable can also be accessed as attributes.
# They are typically used to store the variable's unit, long and standard
# name.
variable_attributes = variable.attrs
print(
    f"One of the variables has the following attributes: {variable_attributes}"
)

# E
# The `xarray.Dataset.isel` method provides a very handy way of accessing
# certain coordinates and variables. Either via keyword arguments
# (i.e. `sel(name=index)`), or by passing a `dict`
# (i.e. `sel({"name": index})`).
level_137 = ds.sel(level=137)

# Even more handy is the access via values instead of indexes using the
# `xarray.Dataset.sel` method. E.g. the time index can be very nicely
# accessed using Python `datetime`.
date = datetime(2020, 1, 1, 1)  # 01.01.2020 at 01:00AM
level_137_01_am = level_137.sel({"time": date})

# Now select the temperature at level 137 and 01.01.2020 01:00AM and plot
# the field.
temperature = level_137_01_am["t"]
temperature.plot()
