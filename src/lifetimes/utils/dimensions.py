import dataclasses
import typing as t

import xarray as xr


@dataclasses.dataclass
class Dimensions:
    """Dimensions of a given dataset."""

    time: int
    x: int
    y: int
    n_vars: t.Optional[int] = None

    @classmethod
    def from_xarray(
        cls, data: t.Union[xr.DataArray, xr.Dataset]
    ) -> "Dimensions":
        """Construct from `xarray` data object."""
        if isinstance(data, xr.DataArray):
            return cls.from_data_array(data)
        elif isinstance(data, xr.Dataset):
            return cls.from_dataset(data)
        raise NotImplementedError(
            f"Cannot construct {cls.__name__} from {type(data)}"
        )

    @classmethod
    def from_data_array(cls, data: xr.DataArray) -> "Dimensions":
        """Construct from `xarray.DataArray`."""
        time, x, y = data.shape
        return cls(
            time=time,
            x=x,
            y=y,
        )

    @classmethod
    def from_dataset(cls, data: xr.Dataset) -> "Dimensions":
        """Construct from `xarray.Dataset`."""
        time, x, y = tuple(data.sizes.values())
        n_vars = len(data.data_vars)
        return cls(
            time=time,
            x=x,
            y=y,
            n_vars=n_vars,
        )

    @property
    def is_multi_variable(self) -> bool:
        """Return whether the shape represents a multi-variable dataset."""
        return self.n_vars is not None

    def to_tuple(self, include_time_dim: bool = True) -> tuple[int, ...]:
        """Return as a tuple."""
        if include_time_dim:
            return self.time, self.x, self.y
        return self.x, self.y

    def to_slices(self) -> t.Iterator[slice]:
        """Return a set of `slice` that allows iterating over a flattened array
        containing multiple variables"""
        flattened_size = self.x * self.y

        if self.n_vars is None:
            yield slice(flattened_size)

        for i in range(self.n_vars):
            start = flattened_size * i
            end = flattened_size * (i + 1)
            yield slice(start, end)
