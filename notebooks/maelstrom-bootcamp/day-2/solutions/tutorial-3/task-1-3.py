import xarray as xr


turbine_cleaned = xr.open_dataset(
    "/p/project/training2223/a6/data/wind_turbine_cleaned.nc"
)

# Resample to an hourly time series and take the mean for each hour.
turbine_resampled = turbine_cleaned.resample(time="1h").mean()

# Remove NaNs that resulted from the resampling.
turbine_resampled_without_nan = turbine_resampled.where(
    turbine_resampled["production"].notnull(), drop=True
)
