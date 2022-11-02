import a6.types as types
import numpy as np
import scipy.ndimage as ndimage


def apply_kernel(
    data: types.Data, kernel: str | np.ndarray, **kwargs
) -> np.ndarray:
    """Apply a given kernel to the data.

    Padding values are filled with the nearest value (`mode="nearest"`).

    """
    if isinstance(kernel, str):
        try:
            factory = _KERNELS[kernel]
        except KeyError:
            raise ValueError(f"Kernel of type '{kernel} not supported")

        kernel = factory(**kwargs)
    return ndimage.convolve(data, kernel, mode="nearest") / kernel.sum()


def create_mean_kernel(size: int) -> np.ndarray:
    """Create a rectangular, normalized kernel for mean.

    Parameters
    ----------
    size : int
        Width and height of the kernel.

    """
    return np.ones((size, size), dtype=np.float32)


def create_gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """Create a gaussian kernel."""
    half_length = (size - 1) / 2.0
    x = np.linspace(-half_length, half_length, size)
    gauss_curve = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gauss_curve, gauss_curve)
    return kernel


_KERNELS = {
    "mean": create_mean_kernel,
    "gaussian": create_gaussian_kernel,
}
