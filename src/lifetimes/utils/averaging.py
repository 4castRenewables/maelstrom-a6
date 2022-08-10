import xarray as xr


def calculate_daily_mean(
    dataset: xr.Dataset, time_coordinate: str = "time"
) -> xr.Dataset:
    """Calculate daily mean for all parameters in a dataset.

    The below implementation works well for both monotonous and non-monotonous
    timeseries data. However, it might not be the most efficient. For mono-
    tonous data, a simple call of `dataset.resample.resample(time='1D').mean()`
    might be more efficient. For non-monotonous data, though, this procedure
    fills the temporal gaps with `NaNs`.

    """
    grouped: xr.Dataset = dataset.groupby(f"{time_coordinate}.date")
    mean: xr.Dataset = grouped.mean()
    # groupy renames the time dimension.
    renamed = mean.rename({"date": time_coordinate})
    # groupy time.date transforms the time values into `datetime.datetime`,
    # which is incompatible with the netCDF format.
    renamed[time_coordinate] = renamed[time_coordinate].astype("datetime64[ms]")
    return renamed
