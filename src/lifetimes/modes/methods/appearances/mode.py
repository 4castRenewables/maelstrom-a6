import dataclasses
import datetime
import functools
import typing as t

import numpy as np
import pandas as pd

Sequence = t.Union[list, tuple, np.ndarray]


@dataclasses.dataclass
class AppearanceIndex:
    """Index of appearance and disappearance of a certain weather regime.

    Parameters
    ----------
    start : int
        Index where the appearance starts.
    end : int
        Index where the appearance ends.

    """

    start: int
    end: int

    @classmethod
    def from_sequences(cls, seq: list[Sequence]) -> list["AppearanceIndex"]:
        """Create from a sequence of indexes.

        The first element in the sequence must represent the first index of
        appearance and the last element the last index of appearance.

        """
        return list(map(cls.from_sequence, seq))

    @classmethod
    def from_sequence(cls, seq: t.Union[list, tuple, np.ndarray]) -> "AppearanceIndex":
        """Create from a sequence of indexes.

        The first element in the sequence must represent the first index of
        appearance and the last element the last index of appearance.

        """
        return cls(start=seq[0], end=seq[-1])

    @property
    def duration(self) -> int:
        """Return the duration of the appearance.

        Since `start` and `end` are indexes, and we want to calculate the
        duration in time units, we need to add 1 to the difference.

        """
        return (self.end - self.start) + 1


@dataclasses.dataclass
class Appearance:
    """Appearance and disappearance of a certain weather regime.

    Parameters
    ----------
    start : datetime.datetime
        Time stamp of the start of the appearance.
    end : datetime.datetime
        Time stamp of the end of the appearance.
    time_delta : datetime.timedelta
        Time delta of the time steps within `start` and `end`.
    index : AppearanceIndex
        Indexes of the respective start/end time stamp in the time series.

    """

    start: datetime.datetime
    end: datetime.datetime
    index: AppearanceIndex = dataclasses.field(repr=False)
    time_delta: datetime.timedelta = dataclasses.field(repr=False)

    @classmethod
    def from_indexes(
        cls, indexes: list[AppearanceIndex], time_series: np.ndarray
    ) -> list["Appearance"]:
        """Create from sequence of `AppearanceIndex`.

        Parameters
        ----------
        indexes : list[AppearanceIndex]
            The indexes of appearance.
        time_series : np.ndarray
            Corresponding time series of the given indexes.

        """
        return [
            cls.from_index(index=index, time_series=time_series) for index in indexes
        ]

    @classmethod
    def from_index(
        cls, index: AppearanceIndex, time_series: np.ndarray
    ) -> "Appearance":
        """Create from `AppearanceIndex`.

        Parameters
        ----------
        index : AppearanceIndex
            The indexes of appearance.
        time_series : np.ndarray
            Corresponding time series of the given indexes.

        """
        start_dt64: np.datetime64 = time_series[index.start]
        end_dt64: np.datetime64 = time_series[index.end]
        time_delta_td64: np.timedelta64 = time_series[1] - time_series[0]
        start = _numpy_datetime64_to_datetime(start_dt64)
        end = _numpy_datetime64_to_datetime(end_dt64)
        time_delta = _numpy_timedelta64_to_timedelta(time_delta_td64)
        return cls(
            start=start,
            end=end,
            index=index,
            time_delta=time_delta,
        )

    @property
    def duration(self) -> datetime.timedelta:
        """Return the duration of the appearance in time units."""
        return (self.end - self.start) + self.time_delta


@dataclasses.dataclass
class Duration:
    """Duration statistics of a mode.

    Parameters
    ----------
    total : datetime.datetime
        Total duration of the mode throughout the time series.
    max : datetime.datetime
        Maximum duration of an appearance.
    min : datetime.datetime
        Minimum duration of an appearance.
    mean : datetime.datetime
        Mean duration of appearance.
    std : datetime.datetime
        Standard deviation of appearance.
    median : datetime.datetime
        Median duration of appearance.

    """

    total: datetime.timedelta
    max: datetime.timedelta
    min: datetime.timedelta
    mean: datetime.timedelta
    std: datetime.timedelta
    median: datetime.timedelta

    @classmethod
    def from_numeric(
        cls, durations: np.ndarray, time_delta: datetime.timedelta
    ) -> "Duration":
        """Construct from an array containing time deltas as numeric values."""
        convert = lambda x: x * time_delta
        total = convert(durations.sum())
        max = convert(np.max(durations))
        min = convert(np.min(durations))
        mean = convert(np.mean(durations))
        std = convert(np.std(durations))
        median = convert(np.percentile(durations, 50))
        return cls(
            total=total,
            max=max,
            min=min,
            mean=mean,
            std=std,
            median=median,
        )


@dataclasses.dataclass
class Statistics:
    """Statistics of a weather mode.

    Parameters
    ----------
    abundance : int
        Total number of appearances over the time period.

    duration_mean : float
        Mean duration of mode appearance.
        Given in time units of the timeseries.
    duration_std : float
        Standard deviation of the mode appearance duration
        Given in time units of the timeseries.

    """

    abundance: int
    duration: Duration

    @classmethod
    def from_appearances(cls, appearances: list[Appearance]) -> "Statistics":
        """Create from a sequence of appearances."""
        time_delta = appearances[0].time_delta
        durations_numeric = np.array(
            [appearance.index.duration for appearance in appearances]
        )
        abundance = durations_numeric.size
        duration = Duration.from_numeric(
            durations=durations_numeric, time_delta=time_delta
        )
        return cls(
            abundance=abundance,
            duration=duration,
        )


@dataclasses.dataclass
class Mode:
    """Properties of a weather mode.

    Parameters
    ----------
    label : int
        Label of the mode.
    appearances : list[Appearance]
        All appearances of the mode.
    statistics : Statistics
        Statistics of the mode appearances.
        E.g. mean duration and standard deviation.

    """

    label: int
    appearances: list[Appearance]
    statistics: Statistics

    @classmethod
    def from_appearances(cls, label: int, appearances: list[Appearance]) -> "Mode":
        """Create from a sequences of appearances."""
        statistics = Statistics.from_appearances(appearances)
        return cls(
            label=label,
            appearances=appearances,
            statistics=statistics,
        )


def _numpy_datetime64_to_datetime(date: np.datetime64) -> datetime.datetime:
    ts = pd.Timestamp(date)
    return ts.to_pydatetime()


def _numpy_timedelta64_to_timedelta(delta: np.timedelta64) -> datetime.timedelta:
    ts = pd.Timedelta(delta)
    return ts.to_pytimedelta()
