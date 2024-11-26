import xarray as xr


turbine = xr.open_dataset("/p/project/training2223/a6/data/wind_turbine.nc")


# Remove outliers by selecting only specific indexes
# and dropping the rest.
turbine_cleaned = turbine.where(
    (
        # A: Find indexes where |P| < 1000 kW
        (abs(turbine["production"]) < 1000)
        &
        # B: and such where P > 0.
        (turbine["production"] > 0)
        &
        # C: and such where P is not NaN.
        turbine["production"].notnull()
    ),
    drop=True,
)
