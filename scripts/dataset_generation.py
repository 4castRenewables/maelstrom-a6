"""Calculate the daily mean in a given set of data source files.

Calculating the daily mean only on each fragment of the dataset (i.e. 00 and 12
model runs) leads to doubled timestamps per day. This is because the daily mean
is calculated for both fragments and then appended to the dataset. Thus, the
daily mean has to be calculated after concatenating all data fragments.

However, calculating the daily mean on a large dataset takes extremely long.
Calculating it on each fragment first and then on the entire dataset is much
more efficient.

As a result, the data are processed in the order:

1. Preprocessing applied to each individual data source file:
   1. Cut off after first 12 hours.
   2. Given levels are selected.
   3. Mean is calculated.
   4. Data Concatenated to the dataset.
2. Postprocessing applied to the entire dataset:
   - Daily mean is calculated.

"""
import pathlib

import a6

path = pathlib.Path("/home/fabian/Documents/MAELSTROM/data/pca")
levels = [500]
outfile = (
    path
    / f"pressure_level_{'_'.join(map(str, levels))}_daily_mean_2017_2020.nc"
)

ds = a6.datasets.EcmwfIfsHres(
    path=path,
    pattern="pl*.nc",
    slice_time_dimension=True,
    preprocessing=a6.datasets.methods.select.select_levels_and_calculate_daily_mean(  # noqa
        levels=levels
    ),
    postprocessing=a6.features.methods.averaging.calculate_daily_mean(),
)

with a6.utils.print_execution_time(description="converting to netcdf"):
    ds.to_xarray().to_netcdf(outfile)
