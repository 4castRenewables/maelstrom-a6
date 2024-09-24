import xarray as xr


weather = xr.open_dataset(
    "/p/project1/training2223/a6/data/ml_level_133_2017_2020.nc"
)

turbine = xr.open_dataset(
    "/p/project1/training2223/a6/data/wind_turbine_cleaned_resampled.nc"
)


def get_closest_grid_point(
    weather_ds: xr.Dataset, turbine_ds: xr.Dataset
) -> xr.Dataset:
    """Get the closest grid point to the wind turbine.

    The weather data have a resolution of 0.1°. Hence,
    we can round the coordinates of the turbine to 1 digit
    and use that grid point's data.

    """

    def _round_coordinate(name: str) -> float:
        return round(float(turbine_ds[name]), 1)

    latitude = _round_coordinate("latitude")
    longitude = _round_coordinate("longitude")
    return weather_ds.sel(latitude=latitude, longitude=longitude)


closest_grid_point_ds = get_closest_grid_point(
    weather_ds=weather, turbine_ds=turbine
)
print(closest_grid_point_ds)
