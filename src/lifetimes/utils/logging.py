import functools
import logging
import sys
import time

logger = logging.getLogger(__name__)


def log_runtime(func):
    """Log the runtime of the given function."""

    @functools.wraps(func)
    def with_logging(*args, **kwargs):
        name = func.__name__
        logger.info(
            "Calling function '%s' with args: %s and kwargs: %s",
            name,
            args,
            kwargs,
        )
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info("Function '%s' was executed in %s seconds", name, duration)
        return result

    return with_logging


def log_to_stdout(level: int = logging.INFO) -> None:
    """Print logs when running in a Jupyter Notebook.

    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level.

    """
    logging.basicConfig(stream=sys.stdout, level=level)
