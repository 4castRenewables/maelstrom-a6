import xarray as xr


ds = xr.open_dataset("/p/project/training2223/a6/data/ml/ml_20200101_00.nc")
print(ds)
