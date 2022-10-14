import multiprocessing


def get_cpu_count() -> int:
    """Return the CPU count."""
    return multiprocessing.cpu_count()
