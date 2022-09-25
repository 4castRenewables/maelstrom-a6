import itertools

import matplotlib.pyplot as plt
import utils
import xarray as xr

data = xr.open_dataset(
    "/p/project/training2223/a6/data/"
    "ml_level_133_2017_2020_wind_turbine_cleaned_resampled.nc"
)


def create_scatter_plot(d: xr.DataArray, x: str, y: str) -> None:
    d.plot.scatter(x=x, y=y)
    plt.show()


combinations = itertools.combinations(data.data_vars, 2)

for x, y in combinations:
    create_scatter_plot(data, x=x, y=y)

# Convert to a pandas.DataFrame and drop the `level` column.
df = data.to_dataframe().drop("level", axis=1)

# Calculate and plot the Pearson correlation coefficients.
corr_matrix = df.corr()
utils.create_heatmap_plot(corr_matrix)
plt.show()

# It actually appears that out of these variables, only the wind speed
# seems to have a valuable correlation.
# Furthermore, it doesn't seem like any of the given variables have a
# strong correlation to one another, except for temperature
# and specific humidity.
# Hence, it might be useful to combine these into a quantity that depends
# on them, such as air density. However, doing so would require a little
# more effort, and there is already a very strong correlation between
# wind speed and production, which already contains very much information
# that will help us train a decent model. Adding air density will very likely
# not improve the model significantly, although it would be nice to investigate
# this.
# TL;DR: We will give it a shot with only the wind speed. However, since
# in reality we will probably not always have the measured wind speed for
# each wind turbine, we will calculate the wind speed from the model data.
