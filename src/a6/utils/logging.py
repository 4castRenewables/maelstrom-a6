import functools
import logging
import os
import sys
import time

import psutil

logger = logging.getLogger(__name__)


def log_consumption(func):
    """Log the consumption of the given function.

    Notes
    -----
    Logs consumption of:
        - memory
        - runtime

    """

    @functools.wraps(func)
    def with_logging(*args, **kwargs):
        logger.debug(
            "Calling function '%s' with args: %s and kwargs: %s",
            func.__name__,
            args,
            kwargs,
        )
        before = _get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        after = _get_process_memory()
        logger.debug(
            (
                "Memory consumption of function '%s': before: %s, after: %s, "
                "consumed: %s, exec time: %s"
            ),
            func.__name__,
            _separate_powers_of_10(before),
            _separate_powers_of_10(after),
            _separate_powers_of_10(after - before),
            _elapsed_time_since(start),
        )
        return result

    return with_logging


def _get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def _elapsed_time_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def _separate_powers_of_10(number: int) -> str:
    return format(number, ",")


def log_to_stdout(level: int = logging.INFO) -> None:
    """Print logs when running in a Jupyter Notebook.

    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level.

    """
    logging.basicConfig(stream=sys.stdout, level=level)


def log_env_vars() -> None:
    env_vars = filter(lambda x: not x[0].startswith("_"), os.environ.items())
    env_vars_sorted = dict(sorted(env_vars))

    env_vars_sorted.pop("MANTIK_USERNAME", None)
    env_vars_sorted.pop("MANTIK_PASSWORD", None)

    logger.info(
        "%s",
        "\n".join(f"{k}: {str(v)}" for k, v in env_vars_sorted.items()),
    )
