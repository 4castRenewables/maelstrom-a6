import logging
import os
import typing as t

import a6.parallel._client as _client
import a6.parallel.types as types
import a6.utils
import ipyparallel.joblib
import joblib

logger = logging.getLogger(__name__)


class IPyParallelClient(_client.Client):
    """Ipyparallel client."""

    _backend: str = "ipyparallel"

    def __init__(
        self,
        ipython_profile: str,
        n_workers: int,
        working_directory: str,
    ):
        """Wait for each engine."""
        self._client = ipyparallel.Client(profile=ipython_profile)
        self._n_workers = n_workers
        self._working_directory = working_directory
        self._engines_awaited = False

    def __enter__(self) -> "IPyParallelClient":
        """Wait for the engines."""
        self._wait_for_engines()
        self._ensure_engines_run_in_working_directory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Shut down the client."""
        self._client.shutdown()

    @property
    def ready(self) -> bool:
        """Return whether the engines are ready."""
        return self._engines_awaited

    @property
    def n_engines(self) -> int:
        """Return the number of engines the client is able to use."""
        if not self.ready:
            raise RuntimeError("Client not ready")
        return len(self._client)

    @property
    def ids(self) -> str:
        """Return the IDs of the client."""
        return str(self._client.ids)

    def _execute(
        self, method: types.Method, arguments: types.Arguments
    ) -> list:
        if not self.ready:
            raise RuntimeError(
                "Client not ready, use context manager to execute"
            )
        logger.info("Starting parallel jobs")
        result = self._execute_jobs_with_engines(method, arguments)
        logger.info("Execution finished")
        return result

    def _ensure_engines_run_in_working_directory(self) -> None:
        self._client[:].map(
            os.chdir, [self._working_directory] * self.n_engines
        )
        logger.info(f"c.ids :{self.ids}")
        balanced_view = self._client.load_balanced_view()
        joblib.register_parallel_backend(
            self._backend,
            lambda: ipyparallel.joblib.IPythonParallelBackend(
                view=balanced_view
            ),
        )

    def _wait_for_engines(self):
        self._client.wait_for_engines(self._n_workers)
        self._engines_awaited = True

    def _execute_jobs_with_engines(
        self, method: types.Method, arguments: types.Arguments
    ) -> list[t.Any]:
        with joblib.parallel_backend("ipyparallel"):
            jobs = (joblib.delayed(method)(*args) for args in arguments)
            engines = joblib.Parallel(n_jobs=self.n_engines)
            result = engines(jobs)
        return result


@a6.utils.log_consumption
def execute_parallel(
    method: types.Method,
    args: types.Arguments,
    ipython_profile: str,
    n_workers: int,
    working_directory: str,
) -> list:
    """Execute a given method parallel using `ipyparallel`.

    Parameters
    ----------
    method : callable
        Method to call.
    args : Iterable[Iterable]
        Arguments to use for each execution.
    ipython_profile : str
        Name of the IPython profile to use.
    n_workers : int
        Number of workers.
    working_directory : str
        Path of the working directory.
        Needed to ensure that all engines are running in the same working
        directory and can access every function and file.
        The logs will also be written in a file in this directory.

    Returns
    -------
    list
        Result of each executed job.

    Examples
    --------
    >>> def add(x, y):
    ...    return x + y
    >>> args = [
    ...    (1, 1),
    ...    (2, 2),
    ...    (3, 3),
    ...]

    >>> execute_parallel(add, args, ...)
    [2, 4, 6]

    """
    # Prepare the engines.
    with IPyParallelClient(
        ipython_profile=ipython_profile,
        n_workers=n_workers,
        working_directory=working_directory,
    ) as client:
        result = client.execute(method, args)
    return result
