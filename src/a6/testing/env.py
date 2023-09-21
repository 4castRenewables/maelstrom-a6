import contextlib
import os
from collections.abc import Iterable
from typing import Any


@contextlib.contextmanager
def env_vars_set(env_vars: dict[str, Any]) -> None:
    """Set the given environment variables and unset afterwards.

    Parameters
    ----------
    env_vars : dict
        Environment variables and values to set.

    Notes
    -----
    All environment variables that were previously set to another value will
    be reset to the initial value afterwards.

    """
    set_env_vars(env_vars)
    yield
    unset_env_vars(env_vars.keys())


def set_env_vars(env_vars: dict[str, Any]) -> None:
    """Set given environment variables.

    Parameters
    ----------
    env_vars : dict
        Environment variables and values to set.

    """
    for key, value in env_vars.items():
        if value is None:
            _unset_env_var(key)
        else:
            os.environ[key] = value


def unset_env_vars(env_vars: Iterable[str]) -> None:
    """Unset given environment variables.

    Parameters
    ----------
    env_vars : iterable
        Environment variables to be unset.

    """
    for key in env_vars:
        _unset_env_var(key)


def _unset_env_var(key: str) -> None:
    try:
        os.environ.pop(key)
    except KeyError:
        # KeyError is raised if variable is already unset.
        pass
