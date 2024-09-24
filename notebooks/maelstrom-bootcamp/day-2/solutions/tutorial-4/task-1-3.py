import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import utils
import xarray as xr

ds = xr.open_dataset(
    "/p/project1/training2223/a6/data/"
    "ml_level_137_temperature_daily_mean_jan_2020.nc"
)

# Reshape the data as before and run PCA retaining 3 components.
matrix = utils.reshape_grid_time_series(ds["t"])
pca = decomposition.PCA(n_components=3).fit(matrix)

# Transform the data into PC space.
transformed = pca.transform(matrix)

# Create the plot
utils.create_3d_scatter_plot(transformed)
plt.show()
