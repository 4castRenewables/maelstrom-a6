import contextlib
import time
import typing as t


@contextlib.contextmanager
def print_execution_time(description: t.Optional[str] = None) -> None:
    """Print time it took to execute the code executed in the context.

    Parameters
    ----------
    description : str

    """
    start = time.time()
    yield
    end = time.time()
    duration = end - start

    if description is None:
        print(f"Execution took {duration} seconds")
    else:
        print(f"Execution ({description}) took {duration} seconds")
