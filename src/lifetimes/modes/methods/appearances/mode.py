import dataclasses
import typing as t

import numpy as np

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
    start : np.datetime64
        Time stamp of the start of the appearance.
    end : np.datetime64
        Time stamp of the end of the appearance.
    index : AppearanceIndex
        Indexes of the respective start/end time stamp in the time series.

    """

    start: np.datetime64
    end: np.datetime64
    index: AppearanceIndex

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
        start: np.datetime64 = time_series[index.start]
        end: np.datetime64 = time_series[index.end]
        return cls(
            start=start,
            end=end,
            index=index,
        )

    @property
    def duration(self) -> int:
        """Return the duration of the appearance in time units."""
        return self.index.duration


@dataclasses.dataclass
class Statistics:
    """Statistics of a weather mode.

    Parameters
    ----------
    total : int
        Total number of appearances over the time period.
    duration_mean : float
        Mean duration of mode appearance.
        Given in time units of the timeseries.
    duration_std : float
        Standard deviation of the mode appearance duration
        Given in time units of the timeseries.

    """

    total: int
    duration_mean: float
    duration_std: float

    @classmethod
    def from_appearances(cls, appearances: list[Appearance]) -> "Statistics":
        """Create from a sequence of appearances."""
        durations = np.array([appearance.duration for appearance in appearances])
        mean = durations.mean()
        std = durations.std()
        return cls(
            total=durations.size,
            duration_mean=mean,
            duration_std=std,
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
