import numpy as np
import xarray as xr

data = xr.open_dataset(
    "/p/project1/training2223/a6/data/"
    "ml_level_133_2017_2020_wind_turbine_cleaned_resampled.nc"
)

data["total_wind_speed"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
