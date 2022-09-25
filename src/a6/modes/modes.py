import typing as t

import xarray as xr

ModesDeterminator = t.Callable[[xr.DataArray], xr.DataArray]


class Modes:
    """A set of modes in a feature."""

    def __init__(self, feature: xr.DataArray):
        self.feature = feature

    def determine_from(self, method: ModesDeterminator) -> xr.DataArray:
        """Determine the modes from a certain method."""
        modes = method(self.feature)
        return modes
