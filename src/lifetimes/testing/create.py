import dataclasses
import datetime
from typing import Optional

from . import data_factories
from . import data_points
from . import fake_datasets
from . import grids
from . import types

DEFAULT_START = datetime.datetime(2000, 1, 1)
DEFAULT_END = datetime.datetime(2000, 2, 28)


@dataclasses.dataclass
class Ellipse:
    """An ellipse on an arbitrary grid.

    Parameters
    ----------
    center : tuple[float, float]
        Center of the ellipsis.
    appearance : str or datetime.datetime
        Date of appearance of the data.
    disappearance : str or datetime.datetime
        Date of disappearance of the data.
    frequency : str, default="1d"
        Frequency of the time steps within the date range.
        E.g., the data may also disappear on an hourly timescale.
    a : float, default=0.15
        Semi-major axis (fractional) of the ellipse (half width).
        Must be between 0 and 1, which represents `a = 0` to
        half the grid size in x-direction (`a = x/2`).
    b : float, default=0.25
        Semi-minor axis (fractional) of the ellipse (half height).
        Must be between 0 and 1, which represents `a = 0` to
        half the grid size in y-direction (`a = y/2`).
    rotate : bool, default False
        Whether to rotate the ellipse by 90 degrees.

    Appears and disappears at given position (centers) and times.

    """

    center: tuple[float, float]
    appearance: types.Timestamp
    disappearance: types.Timestamp
    frequency: str = "1d"
    a: float = 0.15
    b: float = 0.25
    rotate: bool = False


def create_dummy_ecmwf_ifs_hres_dataset(
    grid_size: tuple[int, int] = (10, 10),
    start: types.Timestamp = DEFAULT_START,
    end: types.Timestamp = DEFAULT_END,
    frequency: str = "1d",
    ellipses: Optional[list[Ellipse]] = None,
) -> fake_datasets.FakeEcmwfIfsHresDataset:
    """Create a dummy dataset like ECMWF IFS HRES with elliptical data regions.

    Parameters
    ----------

    """
    rows, columns = grid_size
    grid = grids.TestGrid(rows=rows, columns=columns)

    if ellipses is None:
        ellipses = _create_default_ellipses()

    data = [
        data_points.DataPoints(
            data_factory=_create_elliptical_data_factory(ellipse),
            start=ellipse.appearance,
            end=ellipse.disappearance,
            frequency=ellipse.frequency,
        )
        for ellipse in ellipses
    ]

    return fake_datasets.FakeEcmwfIfsHresDataset(
        grid=grid,
        start=start,
        end=end,
        frequency=frequency,
        data=data,
    )


def _create_default_ellipses() -> list[Ellipse]:
    top_right = (0.5, 0.5)
    bottom_left = (-0.5, -0.5)
    return [
        Ellipse(
            center=top_right,
            appearance=DEFAULT_START,
            disappearance=_add_delay(DEFAULT_START, days=9),
        ),
        Ellipse(
            center=top_right,
            appearance=_add_delay(DEFAULT_START, days=16),
            disappearance=_add_delay(DEFAULT_START, days=26),
        ),
        Ellipse(
            center=bottom_left,
            appearance=DEFAULT_START,
            disappearance=_add_delay(DEFAULT_START, days=12),
        ),
        Ellipse(
            center=bottom_left,
            appearance=_add_delay(DEFAULT_START, days=16),
            disappearance=_add_delay(DEFAULT_START, days=29),
        ),
    ]


def _add_delay(date: datetime.datetime, days: int) -> datetime.datetime:
    return date + datetime.timedelta(days=days)


def _create_elliptical_data_factory(
    ellipse: Ellipse,
) -> data_factories.EllipticalDataFactory:
    return data_factories.EllipticalDataFactory(
        a=ellipse.a,
        b=ellipse.b,
        center=ellipse.center,
        rotate=ellipse.rotate,
    )
