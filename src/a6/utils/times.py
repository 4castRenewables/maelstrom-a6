import datetime

import a6.types as types
import numpy as np
import pandas as pd


def get_time_step_intersection(
    left: types.XarrayData,
    right: types.XarrayData,
    time_coordinate: str,
) -> list[datetime.datetime]:
    """Get the intersection of time steps."""
    # Create sets of the time steps to allow set theory operations.
    left_time_stamps = set(left[time_coordinate].values)
    right_time_stamps = set(right[time_coordinate].values)
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
