import dataclasses
import typing as t

import xarray as xr

XarrayData = t.Union[xr.DataArray, xr.Dataset]


@dataclasses.dataclass
class Dimension:
    """A dimension of a dataset."""

    name: str
    size: int


@dataclasses.dataclass
class TimeDimension:
    """The time dimension of a dataset."""

    name: str
    size: int
    values: xr.DataArray


@dataclasses.dataclass
class Variable:
    """A variable of a dataset."""

    name: str


@dataclasses.dataclass
class Dimensions:
    """Dimensions of a given dataset."""

    time: TimeDimension
    x: Dimension
    y: Dimension
    variables: tuple[Variable]

    @classmethod
    def from_xarray(cls, data: XarrayData, time_dimension: str) -> "Dimensions":
        """Construct from `xarray` data object."""
        time, x, y = _get_temporal_and_spatial_dimension(
            data, time_dimension=time_dimension
        )
        variables = _get_variables(data)
        return cls(
            time=time,
            x=x,
            y=y,
            variables=variables,
        )

    @property
    def is_multi_variable(self) -> bool:
        """Return whether the shape represents a multi-variable dataset."""
        return len(self.variables) > 1

    @property
    def spatial_dimension_names(self) -> tuple[str, str]:
        """Return the names of the sptial dimensions in order (x, y)."""
        return self.x.name, self.y.name

    def to_tuple(self, include_time_dim: bool = True) -> tuple[int, ...]:
        """Return as a tuple."""
        if include_time_dim:
            return self.time.size, self.x.size, self.y.size
        return self.x.size, self.y.size

    def to_variable_name_and_slices(
        self,
    ) -> tuple[t.Iterator[str], t.Iterator[slice]]:
        """Return a set of `slice` that allows iterating over a flattened array
        containing multiple variables"""
        flattened_size = self.x.size * self.y.size

        if not self.is_multi_variable:
            [variable] = self.variables
            yield variable.name, slice(flattened_size)
        else:
            for i, variable in enumerate(self.variables):
                start = flattened_size * i
                end = flattened_size * (i + 1)
                yield variable.name, slice(start, end)


def _get_temporal_and_spatial_dimension(
    data: XarrayData, time_dimension: str
) -> t.Iterator[Dimension]:
    for name, size in data.sizes.items():
        if name == time_dimension:
            yield TimeDimension(
                name=str(name), size=size, values=data[time_dimension]
            )
        else:
            yield Dimension(name=str(name), size=size)


def _get_variables(data: XarrayData) -> tuple[Variable]:
    if isinstance(data, xr.DataArray):
        return (Variable(name=str(data.name)),)
    return tuple(Variable(name=str(var)) for var in data.data_vars)
