import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.datasets.methods as methods


def get_statistics(
    data: xr.Dataset,
    method: methods.normalization.StatisticsMethod,
    levels: list[int],
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> list[float]:
    if len(levels) == 1:
        return [method(data, variable=variable) for variable in data.data_vars]
    return [
        method(data.sel({coordinates.level: level}), variable=variable)
        for level in levels
        for variable in data.data_vars
    ]
