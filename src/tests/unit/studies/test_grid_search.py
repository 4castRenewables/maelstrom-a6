import a6.studies.grid_search as forecast
import sklearn.ensemble as ensemble
import xarray as xr


def test_perform_forecast_model_grid_search(pl_ds):
    model = ensemble.GradientBoostingRegressor
    parameters = {
        "learning_rate": [0.1],
        "n_estimators": [50],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_depth": [3],
    }
    weather = pl_ds.sel(level=500)
    turbine = xr.Dataset(
        data_vars={"production": [i for i in range(49)]},
        coords={
            "time": pl_ds["time"].isel(time=slice(None, 48)),
            "latitude": pl_ds["latitude"].isel(latitude=0),
            "longitude": pl_ds["longitude"].isel(longitude=0),
        },
        attrs={"power rating": "1000", "wind plant": "test-name"},
    )

    forecast.perform_forecast_model_grid_search(
        model=model,
        parameters=parameters,
        weather=weather,
        turbine=turbine,
        log_to_mantik=False,
    )
