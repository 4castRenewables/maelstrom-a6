import datetime
import pathlib
import time

import a6

path = pathlib.Path("/p/largedata2/maelstrom/ap6/ecmwf_era5/nc")
ds = a6.datasets.Era5(path, pattern="*_12.nc").to_xarray()

total = ds["time"].size

for i, step in enumerate(ds["time"]):
    date = datetime.datetime.fromisoformat(str(step.values))
    date_as_str = date.strftime("%Y-%m-%dT%H:%M")
    file = path / f"daily/era5_pl_{date_as_str}.nc"
    if not file.exists():
        print(f"[{round(i / total * 100, 2)}%] Saving {file}")
        sub = ds.sel(time=step).to_netcdf()
        # Sleep 0.1 seconds to avoid crashing due to CPU overload
        time.sleep(0.1)
