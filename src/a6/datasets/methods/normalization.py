import dataclasses
import logging
from collections.abc import Hashable
from collections.abc import Sequence
from typing import Protocol

import xarray as xr

import a6.types as types
import a6.utils as utils

logger = logging.getLogger(__name__)


class StatisticsMethod(Protocol):
    def __call__(self, ds: xr.Dataset, variable: Hashable) -> float:
        pass


@dataclasses.dataclass(frozen=True)
class VariableMinMax:
    name: Hashable
    min: float
    max: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@utils.log_consumption
def calculate_mean(ds: xr.Dataset, variable: Hashable) -> float:
    return float(ds[variable].mean().compute())


@utils.log_consumption
def calculate_std(ds: xr.Dataset, variable: Hashable) -> float:
    return float(ds[variable].std().compute())


@utils.log_consumption
def get_min_max_values(
    data: xr.Dataset, variables: Sequence[Hashable] | None = None
) -> list[VariableMinMax]:
    logger.debug("Getting min and max values for data variables %s", variables)
    variables = variables or list(data.data_vars)
    return [
        VariableMinMax(
            name=variable,
            min=data[variable].min().compute().item(),
            max=data[variable].max().compute().item(),
        )
        for variable in variables
    ]


def min_max_scale_variable(
    data: xr.Dataset, min_max: VariableMinMax
) -> xr.DataArray:
    return min_max_scale(data[min_max.name], min_max=min_max)


def min_max_scale(data: types.DataND, min_max: VariableMinMax) -> types.DataND:
    return (data - min_max.min) / min_max.max
