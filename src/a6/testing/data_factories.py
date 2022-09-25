import abc
import functools
import typing as t

import a6.testing._ellipse as _ellipse
import a6.testing.grids as grids
import a6.testing.types as types
import xarray as xr


class GridDataFactory(abc.ABC):
    """Creates data on a certain grid."""

    @abc.abstractmethod
    def create(self, grid: grids.Grid) -> xr.DataArray:
        """Create the data on a given grid."""


class EllipticalDataFactory(GridDataFactory):
    """Creates elliptical data on a certain grid."""

    def __init__(
        self,
        a: float,
        b: float,
        center: t.Optional[types.Coordinate] = None,
        rotate: bool = False,
    ):
        """Set the ellipse properties.

        Parameters
        ----------
        a : float
            Semi-major axis (fractional) of the ellipse (half width).
            Must be between 0 and 1, which representing `a = 0` to
            half the grid size in x-direction (`a = x/2`).
        b : float
            Semi-minor axis (fractional) of the ellipse (half height).
            Must be between 0 and 1, which representing `a = 0` to
            half the grid size in y-direction (`a = y/2`).
        center : tuple[float, float], optional
            Center of the ellipsis.
            Defaults to `(0.0, 0.0)`
        rotate : bool, default False
            Whether to rotate the ellipse by 90 degrees.

        """
        self.a = a
        self.b = b
        self.center = center or (0.0, 0.0)
        self.rotate = rotate

    @functools.lru_cache
    def create(self, grid: grids.Grid) -> xr.DataArray:
        """Create the elliptical data on the defined grid.

        Parameters
        ----------
        grid : a6.testing.grids.Grid
            The grid on which to create the ellipse.

        Returns
        -------
        xr.DataArray
            Grid with elliptical data.

        """
        return _ellipse.create_ellipse_on_grid(
            grid=grid,
            a=self.a,
            b=self.b,
            center=self.center,
            rotate=self.rotate,
        )
