import xarray as xr

weather = xr.open_dataset(
    "/p/project/training2223/a6/data/"
    "ml_level_133_2017_2020_closest_grid_point.nc"
)
turbine = xr.open_dataset(
    "/p/project/training2223/a6/data/wind_turbine_cleaned_resampled.nc"
)

# Create sets of the time steps to allow set theory operations.
weather_time_stamps = set(weather["time"].values)
turbine_time_stamps = set(turbine["time"].values)

# Get the intersection of both sets.
intersection = weather_time_stamps & turbine_time_stamps

# Python's `set` is an unordered data structure. Hence,
# sort the time values. Passing an unordered sequence
# to `xr.Dataset.sel()` would throw a `InvalidIndexError`.
intersection_sorted = sorted(weather_time_stamps & turbine_time_stamps)

# Select the common dates from each dataset.
weather_sub = weather.sel(time=intersection_sorted)
turbine_sub = turbine.sel(time=intersection_sorted)

# Merge to a single dataset.
# If two datasets with different levels are merged, missing values
# are filled with NaN's. To avoid this, we select the level of each
# dataset and use `compat="override"` to avoid an error due to the
# conflicting level values.
merged = xr.merge(
    [weather_sub.sel(level=133), turbine_sub.sel(level=151.5)],
    compat="override",
)
print(merged)
