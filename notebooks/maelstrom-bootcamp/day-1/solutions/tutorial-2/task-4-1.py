import pandas as pd
import xarray as xr


ds = xr.open_dataset("/p/project1/training2223/a6/data/wind_turbine.nc")

# A
# Get latitude and longitude and look it up on Google Maps or OpenStreetMap.
lat, lon = *ds.coords["latitude"].values, *ds.coords["longitude"].values
print(f"Geological position of the turbine: {lat} {lon}")


# B
print(f"Quantities in the dataset: {list(ds.data_vars)}")

production = ds["production"]
wind_speed = ds["wind_speed"]

print(f"Production data attributes: {production.attrs}")
print(f"Measured wind speed data attributes: {wind_speed.attrs}")

# C
# The native time data type of xarray is `np.datetime64`, which does not
# really give human-readable timedeltas. Thus, we convert to `pd.Timestamp`,
# which gives us a much more comfortable representation.


def calculate_temporal_resolution(
    ds: xr.Dataset, time_coordinate: str = "time"
) -> pd.Timedelta:
    dates = ds[time_coordinate]
    time_0 = pd.Timestamp(dates.isel(time=0).values)
    time_1 = pd.Timestamp(dates.isel(time=1).values)
    return time_1 - time_0


print(f"Temporal resolution: {calculate_temporal_resolution(ds)}")

# D
# Execute the below commands in separate cells, otherwise Jupyter will
# plot both quantities in the same plot and on the same axis, which makes
# it diffucult to see anything for the wind speed.
#
# Note how the production data shows two extreme outliers. Since these are real
# measurement data, they may contain errors.
production.plot()
wind_speed.plot()
