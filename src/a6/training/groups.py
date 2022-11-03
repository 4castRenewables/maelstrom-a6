import dataclasses
import datetime
from collections.abc import Iterator

import a6.types as types
import numpy as np
import xarray as xr


@dataclasses.dataclass
class Groups:
    """Groups in a dataset, grouped by date."""

    def __init__(
        self,
        data: types.XarrayData,
        time_coordinate: str = "time",
        groupby: str = "date",
    ):
        """Get group labels for each date in a time series.

        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The data to get the labels for.
            Must have a time dimension.
        time_coordinate : str, default="time"

        """
        grouped = data.groupby(f"{time_coordinate}.{groupby}")
        self._groups: dict[datetime.datetime, list[int]] = grouped.groups
        self._time = time_coordinate

    @property
    def labels(self) -> list[int]:
        """Return the group labels for the time series.

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
        >>> groups = Groups(da, groupby="date")
        >>> groups.labels
        [1, 1, 2]

        """
        result = []
        for i, (date, indexes) in enumerate(self._groups.items()):
            result.extend(len(indexes) * [i + 1])
        return result

    @property
    def dates(self) -> Iterator[datetime.datetime]:
        """Return the dates."""
        return (datetime.datetime(t.year, t.month, t.day) for t in self._groups)

    def evaluate_cross_validation(
        self, cv: dict, scores: list[str]
    ) -> xr.Dataset:
        """Evaluate the result of a cross validation.

        Returns
        -------
        xr.Dataset
            Contains:
                - scores from the CV
                - groups
                - number of samples per group

        Notes
        -----
        Expects a dict of the form


        """
        result = {
            score: [
                np.mean(cv[f"split{i}_test_{score}"])
                for i, _ in enumerate(self.dates)
            ]
            for score in scores
        }
        return xr.Dataset(
            data_vars={
                score: ([self._time], values)
                for score, values in result.items()
            },
            coords={self._time: list(self.dates)},
        )
