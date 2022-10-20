import a6.types as types


def get_group_labels_for_each_date(
    data: types.XarrayData,
    time_coordinate: str = "time",
) -> list[int]:
    """Get group labels for each date in a time series.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        The data to get the labels for.
        Must have a time dimension.
    time_coordinate : str, default="time"

    Returns
    -------
    list[int]
        List of labels with each date having the same label.

    Examples
    --------
    >>> from datetime import datetime
    >>> import xarray as xr
    >>> dates = [
    ...    datetime(2022, 1, 1, 1),
    ...    datetime(2022, 1, 1, 2),
    ...    datetime(2022, 1, 2, 1)
    ... ]
    >>> da = xr.DataArray(
    ...     [1, 2, 3],
    ...     coords={"time": dates},
    ... )
    >>> get_group_labels_for_each_date(da)
    [1, 1, 2]

    """
    grouped = data.groupby(f"{time_coordinate}.date")

    result = []
    for i, (date, indexes) in enumerate(grouped.groups.items()):
        result.extend(len(indexes) * [i + 1])
    return result
