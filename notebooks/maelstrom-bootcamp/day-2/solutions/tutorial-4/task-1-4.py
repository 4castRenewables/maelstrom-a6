# Uncomment the below magic command before executing the cell.
# %matplotlib notebook
import matplotlib.pyplot as plt
import utils
import xarray as xr

ds = xr.open_dataset(
    "/p/project/training2223/a6/data/"
    "ml_level_137_temperature_daily_mean_jan_2020.nc"
)
transformed = utils.perform_pca_and_transform_into_pc_space(
    ds["t"],
    n_components=3,
)

anim = utils.create_3d_animation_scatter_plot(transformed)  # noqa
plt.show()
