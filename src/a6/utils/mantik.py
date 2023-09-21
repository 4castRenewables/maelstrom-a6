import contextlib
import logging
import os
import pathlib
import shutil
import sys
import tempfile
from collections.abc import Callable

import mlflow


_TRACKING_ENV_VAR = "TRACK_TO_MANTIK"
_CURRENT_EPOCH_ENV_VAR = "CURRENT_EPOCH"
_CPU_USAGE_ENV_VAR = "CPU_USAGE_ENABLED"


@contextlib.contextmanager
def log_logs_as_file(
    level: int = logging.INFO, path: pathlib.Path | None = None
) -> None:
    """Log the logs as a file to mantik (mlflow).

    Parameters
    ----------
    level : int, default=logging.INFO
        The logging level.
    path : pathlib.Path, optional
        Path where to create the temporary log file.

    Raises
    ------
    RuntimeError
        If the temporary log file was never created or deleted before logging
        it to mantik.

    """
    tmp_file = _create_log_file(path)
    _add_file_and_stdout_as_log_stream(level=level, path=tmp_file)

    yield

    _log_file_to_mantik_and_remove_from_local_disk(tmp_file)


def _create_log_file(path: pathlib.Path | None) -> pathlib.Path:
    if path is None:
        path = pathlib.Path(tempfile.mkdtemp())
    return path / "debug.log"


def _add_file_and_stdout_as_log_stream(level: int, path: pathlib.Path) -> None:
    sh = _create_stdout_log_stream(level=level)
    fh = _create_file_log_stream(level=level, path=path)
    logging.basicConfig(
        level=level,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        handlers=[sh, fh],
    )


def _create_stdout_log_stream(level: int) -> logging.Handler:
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    return sh


def _create_file_log_stream(level: int, path: pathlib.Path) -> logging.Handler:
    fh = logging.FileHandler(path)
    fh.setLevel(level)
    return fh


def _log_file_to_mantik_and_remove_from_local_disk(path: pathlib.Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Temporary log file {path} does not exist")

    mlflow.log_artifact(path.as_posix())
    shutil.rmtree(path.parent.as_posix(), ignore_errors=True)


def set_current_epoch(epoch: int) -> None:
    os.environ[_CURRENT_EPOCH_ENV_VAR] = str(epoch)


def get_current_epoch() -> int:
    return int(_get_required_env_var(_CURRENT_EPOCH_ENV_VAR))


def _get_required_env_var(name: str) -> int:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Environment variable {name} unset")
    return int(value)


def call_mlflow_method(func: Callable, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except mlflow.exceptions.MlflowException:
        logging.exception(
            "Calling MLflow method %s with args %s and kwargs %s has failed",
            func,
            args,
            kwargs,
        )
