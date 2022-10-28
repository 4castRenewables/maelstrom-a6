import numpy as np


def create_average_kernel(width: int, height: int | None = None) -> np.ndarray:
    """Create a rectangular, normalized kernel.

    Parameters
    ----------
    width : int
        Width of the kernel.
        Must be an odd number.
    height : int, optional
        Height of the kernel.
        Must be an odd number.
        Defaults to `width`.

    """
    if height is None:
        height = width

    return np.ones((height, width), dtype=np.float64)


def create_gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """Create a gaussian kernel."""
    half_length = (size - 1) / 2.0
    x = np.linspace(-half_length, half_length, size)
    sigma_squared = np.square(sigma)
    gauss_curve = np.exp(-0.5 * np.square(x) / sigma_squared)
    kernel = np.outer(gauss_curve, gauss_curve)
    return kernel
