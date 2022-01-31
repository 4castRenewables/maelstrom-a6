import xarray as xr

from . import modes as _modes


def determine_modes(
    modes: list[_modes.Modes], method: _modes.ModesDeterminator
) -> list[xr.DataArray]:
    """Determine modes with a given method from given features."""
    return [mode.determine_from(method) for mode in modes]
