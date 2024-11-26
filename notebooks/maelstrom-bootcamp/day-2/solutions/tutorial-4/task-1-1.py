import numpy as np
import xarray as xr

ds = xr.open_dataset(
    "/p/project/training2223/a6/data/"
    "ml_level_137_temperature_daily_mean_jan_2020.nc"
)

matrix = np.array(
    # Simply use np.array.flatten() to concatenate each longitude
    # of the dataset.
    [step.values.flatten() for step in ds["t"]]
)

# The matrix now has 31 rows, corresponding to 31 days in the dataset.
# The grid size of our data is 351x551 = 193401. Hence, the resulting matrix
# has 193401 columns, where each column represents the evolution of a grid
# cell over time.
matrix.shape
