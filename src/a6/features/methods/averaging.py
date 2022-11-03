import a6.utils as utils
import xarray as xr


@utils.make_functional
@utils.log_consumption
def calculate_daily_mean(
    dataset: xr.Dataset,
    coordinates: utils.CoordinateNames = utils.CoordinateNames(),
    is_temporally_monotonous: bool = True,
) -> xr.Dataset:
    """Calculate daily mean for all parameters in a dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to calculate the daily mean for.
    coordinates : a6.utils.CoordinateNames, optional
        Name of the coordinates.
    is_temporally_monotonous : bool, default=True
        Whether the dataset is temporally monotonous.

    For temporally monotonous timeseries data, a simple call of
    `dataset.resample(time='1D').mean("time")` is most efficient. For
    non-monotonous datasets, though, this procedure fills the temporal
    gaps with `NaNs`. Hence, there is an alternative implementation for
    temporally non-monotonous datasets.

    """
    if is_temporally_monotonous:
        return dataset.resample({coordinates.time: "1D"}).mean(coordinates.time)

    grouped: xr.Dataset = dataset.groupby(f"{coordinates.time}.date")
    mean: xr.Dataset = grouped.mean()
    # groupy renames the time dimension.
    renamed = mean.rename({"date": coordinates.time})
    # groupy time.date transforms the time values into `datetime.datetime`,
    # which is incompatible with the netCDF format.
    renamed[coordinates.time] = renamed[coordinates.time].astype(
        "datetime64[ms]"
    )
    return renamed
