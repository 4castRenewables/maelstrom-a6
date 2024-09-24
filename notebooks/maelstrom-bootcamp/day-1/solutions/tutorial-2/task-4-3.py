import matplotlib.pyplot as plt
import xarray as xr


ds = xr.open_dataset("/p/project1/training2223/a6/data/wind_turbine.nc")

# A
# Plot wind speed vs. power production on x- and y-axis, respectively.
ds.plot.scatter(x="wind_speed", y="production")
# Set a y-axis limit at 0 kW and 1000 kW due to the outliers.
plt.ylim(0, 1000)

# You can see that the two quantities have a linear correlation.
# This is especially prevalent in the range of v ~= [5, 12] m/s.
# The wings, however, show non-linear relationships. It is also
# visible, that the power production of the wind turbine saturates
# at ~12 m/s.
# The outliers (P(v > 0) = 0 kW) typically result from time windows
# where the wind turbine is switched off, e.g. due to maintenance.
#
# Let's now underline our assumption of a linear correlation with numbers!

# B
# Use xarray's builtin method to compute the Pearson correlation
# coefficient.
pearson_correlation = xr.corr(ds["production"], ds["wind_speed"], dim="time")
print(f"Preason correlation coefficient: {pearson_correlation.values}")

# The Pearson correlation coefficient ranges [-1, 1], where -1 and +1
# indicate a strong negative and positive correlation, respectively.
# Typically, a correlation is considered significant if the coefficient is
# > 0.5 or < -0.5. Here, the value of ~0.91 shows that there is a strong
# positive coupling between the two quantities. More simply put: with great
# wind comes great power!
# Surprise, surprise!
