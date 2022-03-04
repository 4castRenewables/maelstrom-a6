import lifetimes.modes.methods.appearances.mode as mode
import numpy as np
import xarray as xr


def determine_lifetimes_of_modes(
    modes: xr.DataArray,
    time_coordinate: str = "time",
) -> list[mode.Mode]:
    """For a given set of weather modes, calculate their lifetimes.

    Parameters
    ----------
    modes : xr.DataArray
        A timeseries containing the mode labels as a timeseries.
        E.g. if weather modes with labels 1, 2 and 3 exist, a timeseries
        with `N=6` time steps could have the shape `[1, 1, 1, 2, 2, 3]`.
    time_coordinate : str, default="time"
        Name of the time coordinate.
        This is needed to get the time stamps of the appearance and
        disappearance of a weather mode.

    Returns
    -------
    list[lifetimes.modes.methods.lifetimes.Mode]
        Each mode (label) and the respective lifetime statistics.

    Notes
    -----
    The returned statistics about the duration are given in time units of the
    time series. This is because calculating statistics with `np.timedelta64`
    is not possible.

    """
    labels = np.unique(modes)
    time_series = modes[time_coordinate].values
    return [
        _find_all_mode_appearances(
            label=label,
            modes=modes,
            time_series=time_series,
        )
        for label in labels
    ]


def _find_all_mode_appearances(
    label: int, modes: xr.DataArray, time_series: np.ndarray
) -> mode.Mode:
    appearances = _get_mode_appearances(
        label=label,
        modes=modes,
        time_series=time_series,
    )
    return mode.Mode.from_appearances(label=label, appearances=appearances)


def _get_mode_appearances(
    label: int, modes: xr.DataArray, time_series: np.ndarray
) -> list[mode.Appearance]:
    indexes = _get_indexes_of_mode_appearances(label=label, modes=modes)
    appearances_indexes = _find_coherent_appearances_indexes(indexes)
    return mode.Appearance.from_indexes(
        appearances_indexes, time_series=time_series
    )


def _get_indexes_of_mode_appearances(
    label: int, modes: xr.DataArray
) -> np.ndarray:
    [indexes] = np.where(modes == label)
    return indexes


def _find_coherent_appearances_indexes(
    data: np.ndarray,
) -> list[mode.AppearanceIndex]:
    indexes = _get_coherent_groups_of_equivalent_distances(data, distance=1)
    return mode.AppearanceIndex.from_sequences(indexes)


def _get_coherent_groups_of_equivalent_distances(
    data: np.ndarray, distance: int
) -> np.ndarray:
    distances = data[1:] - data[:-1]
    [indexes_of_incoherence] = np.where(distances > distance)
    # Split after each index where a coherent group ends (i.e. at an index of
    # incoherence).
    coherent_groups = np.split(data, indexes_of_incoherence + 1)
    return coherent_groups
