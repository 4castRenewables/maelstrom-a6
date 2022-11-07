import functools
from typing import Any

import a6.types as types


def make_functional(func):
    """Make a function a Functional.

    Notes
    -----
    Functions can still be used as non-functional by passing
    `non_functional=True`.

    Examples
    --------
    >>> import xarray as xr
    >>> @make_functional
    ... def add(data: xr.DataArray, number: int) -> xr.DataArray:
    ...     return data + number
    >>> @make_functional
    ... def multiply(data: xr.DataArray, number: int) -> xr.DataArray:
    ...     return data * number
    >>> da = xr.DataArray(
    ...     [
    ...         [1, 2],
    ...         [3, 4],
    ...     ],
    ... )
    >>> add(1)(da)
    <xarray.DataArray (dim_0: 2, dim_1: 2)>
    array([[2, 3],
           [4, 5]])
    Dimensions without coordinates: dim_0, dim_1
    >>> add(1).apply_to(da)
    <xarray.DataArray (dim_0: 2, dim_1: 2)>
    array([[2, 3],
           [4, 5]])
    Dimensions without coordinates: dim_0, dim_1
    >>> pipe = add(1) >> multiply(4)
    >>> pipe.apply_to(da)
    <xarray.DataArray (dim_0: 2, dim_1: 2)>
    array([[ 8, 12],
           [16, 20]])
    Dimensions without coordinates: dim_0, dim_1
    >>> @make_functional
    ... def subtract_mean(da: xr.DataArray) -> xr.DataArray:
    ...     return da - da.mean()
    >>> subtract_mean().apply_to(da)
    <xarray.DataArray (dim_0: 2, dim_1: 2)>
    array([[-1.5, -0.5],
           [ 0.5,  1.5]])
    Dimensions without coordinates: dim_0, dim_1
    >>> add(da, number=1, non_functional=True)
    <xarray.DataArray (dim_0: 2, dim_1: 2)>
    array([[2, 3],
           [4, 5]])
    Dimensions without coordinates: dim_0, dim_1

    """

    @functools.wraps(func)
    def wrapper(
        *args, non_functional: bool = False, **kwargs
    ) -> Functional | Any:
        if non_functional:
            return func(*args, **kwargs)
        if args:
            raise ValueError(
                "When using a Functional object, arguments have to be provided "
                "with keywords (kwargs)"
            )
        return Functional(func, **kwargs)

    return wrapper


class Functional:
    """Makes a function behave as in a functional programming language.

    Instead of `.` chaining, it allows declaring operations that are supposed
    to be sequentially executed on a data structure using the `>>` operator.

    Examples
    --------
    >>> import xarray as xr
    >>> da = xr.DataArray(
    ...     [
    ...         [1, 2],
    ...         [3, 4],
    ...     ],
    ... )
    >>> def add(ds: xr.DataArray, number: int) -> xr.DataArray:
    ...     return ds + number
    >>> f = Functional(add, 1)
    >>> f.apply_to(da)
    <xarray.DataArray (dim_0: 2, dim_1: 2)>
    array([[2, 3],
           [4, 5]])
    Dimensions without coordinates: dim_0, dim_1

    """

    def __init__(self, func, **kwargs):
        self._func = functools.partial(func, **kwargs)
        self._prev = None
        self._next = None

    def __rshift__(self, operation: "Functional") -> "Functional":
        """Append a subsequent operation"""
        self._next = operation
        self._next._prev = self
        return self._next

    def __call__(self, data: types.DataND) -> types.DataND:
        """Apply the function on the given data."""
        return self.apply_to(data)

    def apply_to(self, data: types.DataND) -> types.DataND:
        """Apply the function on the given data.

        If a following operation was defined, pass the result.

        """
        if self._prev is None:
            return self._func(data)
        return self._func(self._prev.apply_to(data))
