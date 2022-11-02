import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_2d_data(data: np.ndarray) -> plt.Axes:
    """Plot 2D data."""
    return xr.DataArray(data).plot()
