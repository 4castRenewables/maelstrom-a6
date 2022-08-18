import xarray as xr


def calculate_daily_mean(
    dataset: xr.Dataset,
    time_coordinate: str = "time",
    is_temporally_monotonous: bool = True,
) -> xr.Dataset:
    """Calculate daily mean for all parameters in a dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to calculate the daily mean for.
    time_coordinate : str, default="time"
        Name of the time coordinate.
    is_temporally_monotonous : bool, default=True
        Whether the dataset is temporally monotonous.

    For temporally monotonous timeseries data, a simple call of
    `dataset.resample(time='1D').mean("time")` is most efficient. For
    non-monotonous datasets, though, this procedure fills the temporal
    gaps with `NaNs`. Hence, there is an alternative implementation for
    temporally non-monotonous datasets.

    """
    if is_temporally_monotonous:
        return dataset.resample({time_coordinate: "1D"}).mean(time_coordinate)

    grouped: xr.Dataset = dataset.groupby(f"{time_coordinate}.date")
    mean: xr.Dataset = grouped.mean()
    # groupy renames the time dimension.
    renamed = mean.rename({"date": time_coordinate})
    # groupy time.date transforms the time values into `datetime.datetime`,
    # which is incompatible with the netCDF format.
    renamed[time_coordinate] = renamed[time_coordinate].astype("datetime64[ms]")
    return renamed
