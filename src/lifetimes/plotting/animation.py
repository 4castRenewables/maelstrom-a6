import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import xarray as xr


def animate_timeseries(data: xr.DataArray, time_coordinate: str = "time", is_datetime: bool = True, display: bool = True) -> matplotlib.animation.FuncAnimation:
    """Animate time series data.

    Parameters
    ----------
    data : xr.DataArray
    time_coordinate : str, default="time"
        Name of the time coordinate.
    is_datetime : bool, default=True
        Whether the time steps are `datetime.datetime` objects or equivalent.
    display: bool, default=True
        Whether display the animation, i.e. call `plt.show()`.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    """
    if is_datetime:
        times = np.datetime_as_string(data[time_coordinate])
    else:
        times = data[time_coordinate]
    fig = plt.figure()
    im_plot = plt.imshow(data.isel({time_coordinate: 0}).values, vmin=np.min(data), vmax=np.max(data))
    ax = plt.gca()
    title = ax.text(0.5, 1.100, f"time step {times[0]}", transform=ax.transAxes, ha="center")

    def animation(i):
        im_plot.set_data(data.isel({time_coordinate: i}).values)
        title.set_text(f"time step {times[i]}")
        return [im_plot]

    anim = matplotlib.animation.FuncAnimation(fig, animation, frames=len(data[time_coordinate]))
    if display:
        plt.show()
    return anim
