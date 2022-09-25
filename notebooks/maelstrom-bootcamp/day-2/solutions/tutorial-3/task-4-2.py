import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import utils
import xarray as xr

data = xr.open_dataset(
    "/p/project/training2223/a6/data/"
    "ml_level_133_2017_2020_wind_turbine_cleaned_resampled_wind_speed.nc"
)

# We explicitly have to select the coordinates again.
data = data.sel(latitude=52.2, longitude=11.2)

# A
# Just use the last five days as test data, and the rest as train data.
five_days = 5 * 24
train = data.isel(time=slice(None, five_days))
test = data.isel(time=slice(-five_days, -1))


# B
# Here, we use a Random Forest Regressor.
model = ensemble.RandomForestRegressor()


def reshape(d: xr.DataArray) -> np.array:
    """Reshape a row to a column vector.

    The input data have to be reshaped from a row to a column vector
    to be conform with the API of sklearn's `fit` and `predict` method.

    """
    return d.values.reshape(-1, 1)


# Fit the model to the test data.
model = model.fit(X=reshape(train["total_wind_speed"]), y=train["production"])

# C
# Create the forecast with the total wind speed from the NWP model
# test data.
forecast = model.predict(reshape(test["total_wind_speed"]))

# D
# Calculate the MAE.
mae = metrics.mean_absolute_error(y_true=test["production"], y_pred=forecast)
print(f"MAE: {round(mae, 2)} kW")

# E
# Calculate the NMAE.
nmae = mae / 850
print(f"NMAE: {round(100 * nmae, 2)}%")

# F
# Create the plot with the real production and the forecast.
utils.create_production_and_forecast_comparison_plot(
    real=test,
    forecast=forecast,
)
plt.show()
