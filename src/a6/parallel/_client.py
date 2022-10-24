import abc

import a6.parallel.types as types
import a6.utils


class Client(abc.ABC):

    """A client for parallel execution of jobs."""

    @abc.abstractmethod
    def __enter__(self) -> "Client":
        """Initialize the client"""

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Shut down the client."""

    @property
    @abc.abstractmethod
    def ready(self) -> bool:
        """Return whether the client and workers are ready."""

    @a6.utils.log_consumption
    def execute(self, method: types.Method, arguments: types.Arguments) -> list:
        """Execute a method in parallel.

        Parameters
        ----------
        method : Callable
            Method to execute in parallel.
        arguments : Iterable[Iterable]
            Arguments to use for each execution.

        Returns
        -------
        list
            Result of each job.

        Examples
        --------
        >>> import pytest; pytest.skip("")
        >>> def add(x, y):
        ...    return x + y
        >>> args = [
        ...    (1, 1),
        ...    (2, 2),
        ...    (3, 3),
        ... ]
        >>> with Client(...) as client:
        ...     result = client.execute(add, args)
        >>> result
        [2, 4, 6]

        """
        return self._execute(method=method, arguments=arguments)

    @abc.abstractmethod
    def _execute(
        self, method: types.Method, arguments: types.Arguments
    ) -> list:
        ...
