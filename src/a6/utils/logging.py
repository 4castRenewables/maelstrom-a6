import datetime
import functools
import logging
import os
import pathlib
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


class LogFormatter(logging.Formatter):
    def __init__(self, global_rank: int, local_rank: int):
        super().__init__()
        self.start_time = time.time()
        self.rank = global_rank
        self.local_rank = local_rank

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "RANK {} (LOCAL {}) - {} - {} - {}".format(
            self.rank,
            self.local_rank,
            record.levelname,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            datetime.timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return f"{prefix} - {message}" if message else ""


def create_logger(
    global_rank: int,
    local_rank: int,
    verbose: bool = False,
    filepath: pathlib.Path | None = None,
) -> logging.Logger:
    """Adapt the root logging config.

    Use a different log file for each process.

    """
    level = logging.DEBUG if verbose else logging.INFO
    log_formatter = LogFormatter(global_rank=global_rank, local_rank=local_rank)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(log_formatter)

    handlers = [console_handler]

    if filepath is not None:
        path = filepath.with_name(
            f"{filepath.stem}-rank-{global_rank}{filepath.suffix}"
        )
        file_handler = logging.FileHandler(path, "a+")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

        handlers.append(file_handler)

    # create logger and set level to debug
    logging.basicConfig(
        level=level,
        handlers=handlers,
        # Force overriding of existing handlers at runtime.
        force=True,
    )
