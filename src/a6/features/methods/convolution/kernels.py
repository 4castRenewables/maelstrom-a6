import numpy as np


def create_average_kernel(size: int) -> np.ndarray:
    """Create a rectangular, normalized smoothing kernel.

    Parameters
    ----------
    size : int
        Width and height of the kernel.
        Must be an odd number.

    """
    kernel = np.ones((size, size), dtype=np.float64)
    return kernel


def create_gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """Create a gaussian smoothing kernel."""
    half_length = (size - 1) / 2.0
    x = np.linspace(-half_length, half_length, size)
    sigma_squared = np.square(sigma)
    gauss_curve = np.exp(-0.5 * np.square(x) / sigma_squared)
    kernel = np.outer(gauss_curve, gauss_curve)
    return kernel
