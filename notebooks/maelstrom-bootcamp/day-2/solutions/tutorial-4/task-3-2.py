import matplotlib.pyplot as plt
import utils
import xarray as xr

ds = xr.open_dataset(
    "/p/project/training2223/a6/data/"
    "ml_level_137_temperature_daily_mean_2017_2020.nc"
)
hdbscan = utils.perform_hdbscan(
    ds["t"],
    n_components=3,
    min_cluster_size=10,
)
hdbscan.condensed_tree_.plot()
plt.show()
