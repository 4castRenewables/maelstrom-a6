import abc
import functools

import lifetimes.testing.types as types
import numpy as np


class Grid(abc.ABC):
    x_coordinate: str
    y_coordinate: str

    @property
    @functools.lru_cache
    def x_values(self) -> np.ndarray:
        """Return the values in x-direction."""
        return self._x_values

    @property
    @functools.lru_cache
    def y_values(self) -> np.ndarray:
        """Return the values in y-direction."""
        return self._y_values

    @property
    @functools.lru_cache
    def shape(self) -> tuple[int, int]:
        """Return the shape of the spatial grid as (x, y)."""
        return self.x_values.size, self.y_values.size

    @property
    @functools.lru_cache
    def rows(self) -> int:
        """Return the number of rows of the grid."""
        return self.y_values.size

    @property
    @functools.lru_cache
    def columns(self) -> int:
        """Return the number of columns of the grid."""
        return self.x_values.size

    @property
    @abc.abstractmethod
    def _x_values(self) -> np.ndarray:
        ...

    @property
    @abc.abstractmethod
    def _y_values(self) -> np.ndarray:
        ...

    @property
    def coordinates(self) -> list[str]:
        """Return the dimension names."""
        return [self.y_coordinate, self.x_coordinate]

    @property
    @functools.lru_cache
    def xarray_coords_dict(self) -> types.CoordinateDict:
        """Return the coordinates as confirm to xarray's `coords` kwarg.

        The y- and x-coordinate come first and second, respectively, to
        follow the CF 1.6 conventions (see
        https://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#dimensions
        ).

        """
        return {
            self.x_coordinate: self.x_values,
            self.y_coordinate: self.y_values,
        }


class EcmwfIfsHresGrid(Grid):
    """Represents the ECMWF IFS HRES data."""

    x_coordinate = "longitude"
    y_coordinate = "latitude"

    @property
    @functools.lru_cache
    def _x_values(self) -> np.ndarray:
        return np.arange(-25.0, 30.0, 0.1)

    @property
    @functools.lru_cache
    def _y_values(self) -> np.ndarray:
        return np.arange(30.0, 75.0, 0.1)


class TestGrid(Grid):
    """A testing grid of a given size."""

    x_coordinate = "lon"
    y_coordinate = "lat"

    def __init__(self, rows: int = 1, columns: int = 1):
        self._rows = rows
        self._columns = columns

    @property
    @functools.lru_cache
    def _x_values(self) -> np.ndarray:
        return np.arange(0.0, float(self._columns))

    @property
    @functools.lru_cache
    def _y_values(self) -> np.ndarray:
        return np.arange(0.0, float(self._rows))
