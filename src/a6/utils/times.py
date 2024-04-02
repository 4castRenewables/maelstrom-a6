import datetime

import numpy as np
import pandas as pd

import a6.datasets.coordinates as _coordinates
import a6.types as types


def get_time_step_intersection(
    left: types.XarrayData,
    right: types.XarrayData,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> list[datetime.datetime]:
    """Get the intersection of time steps."""
    # Create sets of the time steps to allow set theory operations.
    left_time_stamps = set(left[coordinates.time].values)
    right_time_stamps = set(right[coordinates.time].values)
    intersection = left_time_stamps & right_time_stamps
    return sorted(intersection)


def time_steps_as_dates(
    data: types.XarrayData,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> list[datetime.datetime]:
    """Convert the time steps to `datetime.datetime` with YYYY-MM-DD."""
    return [datetime.datetime(d.year, d.month, d.day) for d in time_steps_as_datetimes(data)]


def time_steps_as_datetimes(
    data: types.XarrayData,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> list[datetime.datetime]:
    """Convert the time steps to `datetime.datetime`."""
    return [numpy_datetime64_to_datetime(s) for s in data[coordinates.time].values]


def numpy_datetime64_to_datetime(date: np.datetime64) -> datetime.datetime:
    """Convert numpy.datetime64 to Python's datetime."""
    ts = pd.Timestamp(date)
    return ts.to_pydatetime()


def numpy_timedelta64_to_timedelta(
    delta: np.timedelta64,
) -> datetime.timedelta:
    """Convert numpy.timedelta64 to Python's datetime."""
    ts = pd.Timedelta(delta)
    return ts.to_pytimedelta()
