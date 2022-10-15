import contextlib
import typing as t


def expect_raise_if_exception(
    expected: t.Any,
) -> contextlib.AbstractContextManager:
    """Create a context that expects a raised exception or no raised exception.

    Parameters
    ----------
    expected
        The expected result.

    Returns
    -------
    _pytest.python_api.RaisesContext
        If expected is of type `Exception`
    contextlib.nullcontext
        otherwise.

    """
    import pytest

    return (
        pytest.raises(type(expected))
        if isinstance(expected, Exception)
        else contextlib.nullcontext()
    )
