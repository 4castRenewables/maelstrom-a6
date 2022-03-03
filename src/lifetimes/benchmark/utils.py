import logging
import sys
import typing as t

import dask.delayed


def wrap_benchmark_method_with_logging(method: t.Callable) -> t.Callable:
    """
    Wrap benchmark method by adding logging setup.

    Parameters
    ----------
    method: Method to wrap with logging

    Returns
    -------
        Callable.

    Notes
    -----
    dask.distributed.delayed must be handed a single function; thus logging must be
    configured in the wrapped function call. See `make_method_lazy` function.

    """

    def wrapped(*args, **kwargs) -> t.Callable:
        logger = logging.getLogger(__name__)

        logging.basicConfig(
            stream=sys.stdout,
            filemode="w",
            level=logging.DEBUG,
        )
        logger.info("Starting benchmark execution.")
        return method(*args, **kwargs)

    return wrapped


def make_method_lazy(
    benchmark_method: t.Callable,
) -> t.Callable:
    """
    Lazily execute the benchmark method.

    Parameters
    ----------
    benchmark_method: Callable to be benchmarked.
    benchmark_method_args: Arguments for benchmark method.
    log_directory: Log directory.
    job_name: Job name

    Returns
    -------

    """
    return dask.delayed(benchmark_method)
