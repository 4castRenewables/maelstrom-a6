import a6.studies.grid_search as forecast
import xarray as xr


def test_perform_forecast_model_grid_search(pl_ds):
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
        weather=weather,
        turbine=turbine,
        log_to_mantik=False,
    )
