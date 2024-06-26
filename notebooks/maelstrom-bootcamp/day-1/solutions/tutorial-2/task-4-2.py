import matplotlib.pyplot as plt
import utils
import xarray as xr


ds = xr.open_dataset("/p/project1/training2223/a6/data/wind_turbine.nc")

# Get a subset of the first 60 time steps in the timeseries.
ds_subset = ds.isel(time=slice(None, 60))

utils.create_twin_y_axis_plot(ds_subset, left="production", right="wind_speed")

plt.show()
# From eyeball analysis, there seems to be a strong correlation between
# power production and measured wind speed.
