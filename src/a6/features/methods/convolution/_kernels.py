import numpy as np


def create_mean_kernel(size: int) -> np.ndarray:
    """Create a rectangular, normalized kernel for mean.

    Parameters
    ----------
    size : int
        Width and height of the kernel.

    """
    _check_size_is_odd(size)
    return np.ones((size, size), dtype=np.float32)


def create_gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """Create a gaussian kernel."""
    _check_size_is_odd(size)
    half_length = (size - 1) / 2.0
    x = np.linspace(-half_length, half_length, size)
    gauss_curve = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gauss_curve, gauss_curve)
    return kernel


def _check_size_is_odd(size: int) -> None:
    if size % 2 == 0:
        raise ValueError("Size of kernel must an odd number")
