import dataclasses
import datetime
import typing as t

import numpy as np
import pandas as pd

Sequence = t.Union[list, tuple, np.ndarray]


@dataclasses.dataclass
class AppearanceIndex:
    """Index of appearance and disappearance of a certain weather regime.

    Parameters
    ----------
    label : int
        Label of the respective mode.
    start : int
        Index where the appearance starts.
    end : int
        Index where the appearance ends.

    """

    label: int
    start: int
    end: int

    @classmethod
    def from_sequences(
        cls, label: int, seq: list[Sequence]
    ) -> list["AppearanceIndex"]:
        """Create from a sequence of indexes.

        The first element in the sequence must represent the first index of
        appearance and the last element the last index of appearance.

        """
        return [cls.from_sequence(label=label, seq=s) for s in seq]

    @classmethod
    def from_sequence(
        cls, label: int, seq: t.Union[list, tuple, np.ndarray]
    ) -> "AppearanceIndex":
        """Create from a sequence of indexes.

        The first element in the sequence must represent the first index of
        appearance and the last element the last index of appearance.

        """
        return cls(label=label, start=seq[0], end=seq[-1])

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
    label : int
        Label of the respective mode.
    start : datetime.datetime
        Time stamp of the start of the appearance.
    end : datetime.datetime
        Time stamp of the end of the appearance.
    time_delta : datetime.timedelta
        Time delta of the time steps within `start` and `end`.
    index : AppearanceIndex
        Indexes of the respective start/end time stamp in the time series.

    """

    label: int
    start: datetime.datetime
    end: datetime.datetime
    index: AppearanceIndex = dataclasses.field(repr=False)
    time_delta: datetime.timedelta = dataclasses.field(repr=False)

    @classmethod
    def from_indexes(
        cls, label: int, indexes: list[AppearanceIndex], time_series: np.ndarray
    ) -> list["Appearance"]:
        """Create from sequence of `AppearanceIndex`.

        Parameters
        ----------
        label : int
            Label of the respective mode.
        indexes : list[AppearanceIndex]
            The indexes of appearance.
        time_series : np.ndarray
            Corresponding time series of the given indexes.

        """
        return [
            cls.from_index(label=label, index=index, time_series=time_series)
            for index in indexes
        ]

    @classmethod
    def from_index(
        cls, label: int, index: AppearanceIndex, time_series: np.ndarray
    ) -> "Appearance":
        """Create from `AppearanceIndex`.

        Parameters
        ----------
        label : int
            Label of the respective mode.
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
            label=label,
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
    label : int
        Label of the respective mode.
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

    label: int
    total: datetime.timedelta
    max: datetime.timedelta
    min: datetime.timedelta
    mean: datetime.timedelta
    std: datetime.timedelta
    median: datetime.timedelta

    @classmethod
    def from_numeric(
        cls, label: int, durations: np.ndarray, time_delta: datetime.timedelta
    ) -> "Duration":
        """Construct from an array containing time deltas as numeric values."""

        def convert(absolute: float) -> datetime.timedelta:
            return absolute * time_delta

        total = convert(durations.sum())
        max = convert(np.max(durations))
        min = convert(np.min(durations))
        mean = convert(np.mean(durations))
        std = convert(np.std(durations))
        median = convert(np.percentile(durations, 50))
        return cls(
            label=label,
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
    label : int
        Label of the respective mode.
    abundance : int
        Total number of appearances over the time period.
    duration_mean : float
        Mean duration of mode appearance.
        Given in time units of the timeseries.
    duration_std : float
        Standard deviation of the mode appearance duration
        Given in time units of the timeseries.

    """

    label: int
    abundance: int
    duration: Duration

    @classmethod
    def from_appearances(
        cls, label: int, appearances: list[Appearance]
    ) -> "Statistics":
        """Create from a sequence of appearances."""
        time_delta = appearances[0].time_delta
        durations_numeric = np.array(
            [appearance.index.duration for appearance in appearances]
        )
        abundance = durations_numeric.size
        duration = Duration.from_numeric(
            label=label, durations=durations_numeric, time_delta=time_delta
        )
        return cls(
            label=label,
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
    def from_appearances(
        cls, label: int, appearances: list[Appearance]
    ) -> "Mode":
        """Create from a sequences of appearances."""
        statistics = Statistics.from_appearances(
            label=label, appearances=appearances
        )
        return cls(
            label=label,
            appearances=appearances,
            statistics=statistics,
        )

    def get_dates(
        self,
        start: t.Optional[datetime.datetime] = None,
        end: t.Optional[datetime.datetime] = None,
    ) -> t.Iterator[datetime.datetime]:
        """Return all dates of the appearance of the mode.

        Parameters
        ----------
        start : datetime.datetime
            Start of the dates.
        start : datetime.datetime
            End of the dates.

        Returns
        -------
        t.Iterator[datetime.datetime]
            All dates where this mode appeared.
            Only within `start` and `end` if given.

        """

        def date_within_range(d):
            if start is None and end is None:
                return True
            elif start is not None and end is None:
                return start <= d
            elif start is None and end is not None:
                return d <= end
            else:
                return start <= d <= end

        return (
            date.to_pydatetime()
            for appearance in self.appearances
            for date in pd.date_range(
                appearance.start, appearance.end, freq="1d"
            )
            if date_within_range(date)
        )


def _numpy_datetime64_to_datetime(date: np.datetime64) -> datetime.datetime:
    ts = pd.Timestamp(date)
    return ts.to_pydatetime()


def _numpy_timedelta64_to_timedelta(
    delta: np.timedelta64,
) -> datetime.timedelta:
    ts = pd.Timedelta(delta)
    return ts.to_pytimedelta()
