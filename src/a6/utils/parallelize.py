import concurrent.futures
import itertools
import logging
import multiprocessing
from collections.abc import Callable
from collections.abc import Iterable

logger = logging.getLogger(__name__)


def parallelize_with_multiprocessing(
    function: Callable,
    args_zipped: Iterable,
    single_arg: bool = False,
    processes: int = -1,
    kwargs_as_dict: dict | None = None,
):
    """Parallelize function with args provided in zipped format.

    Parameters
    ----------
    function: function to parallelize
    args_zipped : Iterable
        Args of function in zipped format.
    single_arg : bool, default=False
        If `args_zipped` is an iterable of just one argument.
    processes : int, default=-1
        Number of processed to use.
        If `-1`, use number of available threads.

    Returns
    -------
    function applied to args

    """
    if processes == 1:
        results = []
        if kwargs_as_dict is None:
            kwargs_as_dict = {}
        for args in args_zipped:
            if isinstance(args, str):
                args = (args,)
            results.append(function(*args, **kwargs_as_dict))
        return results

    processes = _get_number_of_processes(processes)

    logger.debug("Number of processes: %s", processes)

    if single_arg and kwargs_as_dict is None:
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(function, args_zipped)
    else:
        if single_arg:
            args_zipped = ((arg,) for arg in args_zipped)

        if kwargs_as_dict is not None:
            kwargs_iter = itertools.repeat(kwargs_as_dict)
        else:
            kwargs_iter = itertools.repeat({})

        with multiprocessing.Pool(processes=processes) as pool:
            results = _starmap_with_kwargs(
                pool, function, args_zipped, kwargs_iter
            )

    return results


def _get_number_of_processes(processes: int):
    if processes < 0:
        processes = multiprocessing.cpu_count() + processes + 1
    return processes


def _starmap_with_kwargs(
    pool, function: Callable, args_iter: Iterable, kwargs_iter: Iterable
):
    """Parallelize functions with args and kwargs."""
    if kwargs_iter is None:
        args_for_starmap = zip(itertools.repeat(function), args_iter)
    else:
        args_for_starmap = zip(
            itertools.repeat(function), args_iter, kwargs_iter
        )
    return pool.starmap(_apply_args_and_kwargs, args_for_starmap)


def _apply_args_and_kwargs(fn: Callable, args, kwargs):
    """Parallelize functions with args and kwargs."""
    return fn(*args, **kwargs)

def parallelize_with_futures(
    func: Callable,
    kwargs: list[dict],
) -> list:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(func, **d): i for i, d in enumerate(kwargs)}

        results = []

        for future in concurrent.futures.as_completed(futures):
            index = futures[future]

            try:
                result = future.result()
            except Exception as e:
                raise RuntimeError(
                    f"Function call #{index} has failed with input {kwargs[index]}"
                ) from e
            else:
                results.append(result)

    return results
