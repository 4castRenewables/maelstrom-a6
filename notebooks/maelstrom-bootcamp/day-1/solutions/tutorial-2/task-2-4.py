import pandas as pd
import xarray as xr


ds = xr.open_dataset("/p/project/training2223/a6/data/ml/ml_20200101_00.nc")
ds2 = xr.open_dataset("/p/project/training2223/a6/data/ml/ml_20200101_12.nc")

print(f"The second dataset's time coordinate: {ds2['time']}")


def get_time_intersection(left: xr.DataArray, right: xr.DataArray) -> pd.Index:
    left_index = pd.Index(left)
    right_index = pd.Index(right)
    return left_index.intersection(right_index)


print("The datasets contain overlapping time steps!\n")
print(get_time_intersection(ds["time"], ds2["time"]))
