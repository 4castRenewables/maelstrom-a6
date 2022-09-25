import matplotlib.pyplot as plt
import xarray as xr


turbine = xr.open_dataset("/p/project/training2223/a6/data/wind_turbine.nc")
turbine_cleaned = xr.open_dataset(
    "/p/project/training2223/a6/data/wind_turbine_cleaned.nc"
)

# A
# Inspect the result of the data cleaning.

# Via a scatter plot.
turbine.plot.scatter(x="wind_speed", y="production", color="grey")
turbine_cleaned.plot.scatter(x="wind_speed", y="production")
plt.ylim(0, 1000)

# Or via histograms
# bins = range(0, 1000, 2)
# turbine["production"].plot.hist(bins=bins)
# turbine_cleaned["production"].plot.hist(bins=bins)

# B
# Calculate the fraction of data that was lost using the dataset size
# before and after.
fraction_lost = (
    1 - turbine_cleaned["production"].size / turbine["production"].size
)
percentage_lost = round(100 * fraction_lost, 2)
print(
    (
        f"{round(percentage_lost, 2)}% of the original data were removed by "
        "the cleaning procedure"
    )
)
