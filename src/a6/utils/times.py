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
