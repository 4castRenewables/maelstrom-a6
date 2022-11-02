import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_2d_data(data: np.ndarray, flip: bool = False) -> plt.Axes:
    """Plot 2D data."""
    if flip:
        data = np.flip(data, axis=0)
    return xr.DataArray(data).plot()
