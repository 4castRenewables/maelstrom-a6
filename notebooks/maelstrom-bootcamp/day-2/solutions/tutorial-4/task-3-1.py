import hdbscan as _hdbscan
import matplotlib.pyplot as plt
import utils
import xarray as xr

ds = xr.open_dataset(
    "/p/project1/training2223/a6/data/"
    "ml_level_137_temperature_daily_mean_2017_2020.nc"
)

# Perform the PCA and transform the data into PC space.
transformed = utils.perform_pca_and_transform_into_pc_space(
    ds["t"],
    n_components=3,
)

# A
# Apply HDBSCAN on the transformed data.
hdbscan = _hdbscan.HDBSCAN(min_cluster_size=10).fit(transformed)

# B
# Plot the time series in PC space with the clusters colored.
# Use the cluster labels als color codes.
utils.create_3d_scatter_plot(
    transformed,
    colors=hdbscan.labels_,
)
plt.show()

# C
# Plot the labels as a time series and color each point accordingly.
utils.create_label_time_series_scatter_plot(
    data=ds,
    clusters=hdbscan,
)

plt.show()
