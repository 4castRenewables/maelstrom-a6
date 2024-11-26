import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import utils
import xarray as xr

ds = xr.open_dataset(
    "/p/project/training2223/a6/data/"
    "ml_level_137_temperature_daily_mean_jan_2020.nc"
)

matrix = utils.reshape_grid_time_series(ds["t"])

# A
# Simply initialize the PCA and directly fit the data.
pca = decomposition.PCA().fit(matrix)

# B
utils.create_scree_test_plot(pca)
plt.show()

# Note that the number of PCs corresponds to the number of days in the
# dataset (31 days in January).
# The plot shows that a significant amount of the dataset's variance
# (~90%) is explained with the first 11 PCs. We can now use these
# to transform our data in a 11-dimensional space that represents
# the phase space of our dynamical weather system that our time series
# temperature data reflect. If we are happy with less than this, we might
# even choose fewer PCs for the next step.

# C
# Plot the first 3 PCs.
for i in range(3):
    pc = utils.restore_original_grid_shape(
        original=ds, reshape=pca.components_[i]
    )
    pc.plot()
    plt.show()
