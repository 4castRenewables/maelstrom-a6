import lifetimes.testing.grids as grids
import lifetimes.testing.types as types
import numpy as np
import xarray as xr


def create_ellipse_on_grid(
    grid: grids.Grid,
    a: float,
    b: float,
    center: types.Coordinate,
    rotate: bool = False,
) -> xr.DataArray:
    """Create elliptic data on a given grid.

    Parameters
    ----------
    grid : lifetimes.testing.grids.Grid
        The grid on which to create the ellipse.
    a : float
        Semi-major axis (fractional) of the ellipse (half width).
        Must be between 0 and 1, i.e. `a = 0` and `a = diameter_x / 2`,
        where `diameter_x` is the diameter of the grid in x-direction.
    b : float
        Semi-minor axis (fractional) of the ellipse (half height).
        Must be between 0 and 1, i.e. `a = 0` and `a = diameter_y / 2`,
        where `diameter_x` is the diameter of the grid in y-direction.
    center : tuple[float, float]
        Center of the ellipsis.
    rotate : bool, default False
        Whether to rotate the ellipse by 90 degrees.

    Returns
    -------
    xr.DataArray
        DataArray of shape `size` with an "ellipse" centered at `center`.
        Data inside

    """
    data = xr.DataArray(
        data=0.0,
        dims=grid.coordinates,
        coords=grid.xarray_coords_dict,
    )
    inside_ellipse = _determine_grid_points_inside_ellipse(
        rows=grid.rows,
        columns=grid.columns,
        a=a,
        b=b,
        center=center,
    )
    data.data[inside_ellipse] = 1.0
    if rotate:
        data.data = np.rot90(data.data)
    return data


def _determine_grid_points_inside_ellipse(
    rows: int, columns: int, a: float, b: float, center: types.Coordinate
) -> np.ndarray:
    x = _create_range_around_origin(columns)
    y = _create_range_around_origin(rows, as_columns=True)
    center_shift = _calculate_center_shift(
        diameter_x=columns, diameter_y=rows, center=center
    )
    inside_ellipse = _identify_data_points_inside_ellipse(
        x=x,
        y=y,
        a_frac=a,
        b_frac=b,
        center=center_shift,
    )
    return inside_ellipse


def _create_range_around_origin(
    size: int, as_columns: bool = False
) -> np.ndarray:
    a_range = np.arange(-np.floor(size / 2), np.ceil(size / 2))
    if as_columns:
        return a_range[::-1, None]
    return a_range


def _calculate_center_shift(
    diameter_x: int, diameter_y: int, center: types.Coordinate
) -> types.Coordinate:
    x0_frac, y0_frac = center
    radius_x, radius_y = diameter_x / 2, diameter_y / 2
    x0 = _fraction_of_radius_to_absolute(x0_frac, radius=radius_x)
    y0 = _fraction_of_radius_to_absolute(y0_frac, radius=radius_y)
    return x0, y0


def _identify_data_points_inside_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    a_frac: float,
    b_frac: float,
    center: types.Coordinate,
) -> np.ndarray:
    x0, y0 = center
    radius_x, radius_y = x.size / 2, y.size / 2
    a = _fraction_of_radius_to_absolute(a_frac, radius=radius_x)
    b = _fraction_of_radius_to_absolute(b_frac, radius=radius_y)
    inside_ellipse = ((x - x0) / a) ** 2 + ((y - y0) / b) ** 2 <= 1
    return inside_ellipse


def _fraction_of_radius_to_absolute(frac: float, radius: float) -> float:
    """Return the fraction of the radius as an absolute number.

    Given the total size (diameter) of the grid and a fraction [0;1],
    calculates the fraction of the radius of the grid as an absolute value.

    E.g. if the grid has size 5x5, the radius is 2.5 indexes. Given a fraction
    `frac = 0.5`, a circle (or ellipse with `a = b`) is supposed to have a
    radius half the grid radius, hence `abs = 0.5 * 2.5 = 1.25`.

    This absolute, even though it represents the radius (semi-major/minor axis)
    in terms of indexes, does not have to be rounded since the ellipse equation
    is used instead of index slicing.

    """
    return frac * radius
