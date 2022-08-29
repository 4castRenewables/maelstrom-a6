import dataclasses
import logging
import typing as t

import lifetimes.utils.coordinates as _coordinates
import xarray as xr

logger = logging.getLogger(__name__)

XarrayData = t.Union[xr.DataArray, xr.Dataset]


@dataclasses.dataclass(frozen=True)
class Dimension:
    """A dimension of a dataset."""

    name: str
    size: int


@dataclasses.dataclass(frozen=True)
class SpatialDimension(Dimension):
    """A spatial dimension of a dataset."""


@dataclasses.dataclass(frozen=True)
class TemporalDimension(Dimension):
    """The time dimension of a dataset."""

    name: str
    size: int
    values: xr.DataArray


@dataclasses.dataclass(frozen=True)
class Variable:
    """A variable of a dataset."""

    name: str


@dataclasses.dataclass(frozen=True)
class _Dimensions:
    time: TemporalDimension
    latitude: SpatialDimension
    longitude: SpatialDimension


@dataclasses.dataclass(frozen=True)
class SpatioTemporalDimensions:
    """SpatioTemporalDimensions of a given dataset.

    Notes
    -----
    According to CF 1.6 conventions, the order of dimensions is
    T, Z (level), Y (latitude), X (longitude). When accessing `data.sizes` of
    an xarray data object, it will be returned as
    `Frozen({'x': 10, 'y': 10, 'time': 5})`.

    """

    time: TemporalDimension
    y: SpatialDimension
    x: SpatialDimension
    variables: tuple[Variable]

    @classmethod
    def from_xarray(
        cls,
        data: XarrayData,
        coordinates: _coordinates.CoordinateNames,
    ) -> "SpatioTemporalDimensions":
        """Construct from `xarray` data object."""
        dimensions = _get_temporal_and_spatial_dimension(
            data, coordinates=coordinates
        )
        logger.info("Got dimensions %s", dimensions)
        variables = _get_variables(data)
        return cls(
            time=dimensions.time,
            y=dimensions.latitude,
            x=dimensions.longitude,
            variables=variables,
        )

    @property
    def is_multi_variable(self) -> bool:
        """Return whether the shape represents a multi-variable dataset."""
        return len(self.variables) > 1

    @property
    def spatial_dimension_names(self) -> tuple[str, str]:
        """Return the names of the spatial dimensions in order (y, x)."""
        return self.y.name, self.x.name

    @property
    def variable_names(self) -> t.Iterator[str]:
        """Return the names of the variables."""
        return (variable.name for variable in self.variables)

    def shape(self, include_time_dim: bool = True) -> tuple[int, ...]:
        """Return the shape as a tuple."""
        if include_time_dim:
            return self.time.size, self.y.size, self.x.size
        return self.y.size, self.x.size

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
    data: XarrayData, coordinates: _coordinates.CoordinateNames
) -> _Dimensions:
    return _Dimensions(
        time=TemporalDimension(
            name=coordinates.time,
            size=data.coords[coordinates.time].size,
            values=data[coordinates.time],
        ),
        latitude=SpatialDimension(
            name=coordinates.latitude,
            size=data.coords[coordinates.latitude].size,
        ),
        longitude=SpatialDimension(
            name=coordinates.longitude,
            size=data.coords[coordinates.longitude].size,
        ),
    )


def _get_variables(data: XarrayData) -> tuple[Variable]:
    if isinstance(data, xr.DataArray):
        return (Variable(name=str(data.name)),)
    return tuple(Variable(name=str(var)) for var in data.data_vars)
