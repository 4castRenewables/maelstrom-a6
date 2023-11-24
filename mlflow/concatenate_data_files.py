import argparse
import logging
import pathlib
import time

import xarray as xr

import a6

a6.utils.logging.log_to_stdout()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-path",
    type=pathlib.Path,
)
parser.add_argument(
    "--pattern",
    type=str,
)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    start = time.time()

    args = parser.parse_args()

    path: pathlib.Path = args.data_path
    patterns: list[str] = args.pattern.split(",")

    logger.info(
        "Reading data files from path %s with patterns %s", path, patterns
    )

    for pattern in patterns:
        outfile = path / f"../ecmwf_ifs_{pattern}_full.nc"

        logger.info(
            "Reading files with pattern %s (outfile=%s)", pattern, outfile
        )

        files = sorted(list(path.rglob(f"{pattern}_*.nc")))
        logger.info("Reading from %i files", len(files))

        coordinates = a6.datasets.coordinates.Coordinates()
        preprocessing = a6.datasets.methods.slicing.slice_dataset(
            dimension=coordinates.time,
            slice_until=12,
        )

        logger.info("Reading dataset files")
        start_reading = time.time()

        ds = xr.open_mfdataset(
            files,
            engine="netcdf4",
            concat_dim="time",
            combine="nested",
            coords="minimal",
            data_vars="minimal",
            preprocess=preprocessing,
            compat="override",
            parallel=False,
            drop_variables=None,
        )

        logger.info(
            "Finished reading in %s seconds", time.time() - start_reading
        )

        start_writing = time.time()
        logger.info("Writing to netCDF")

        ds.to_netcdf(outfile)

        logger.info(
            "Finished writing in %s seconds", time.time() - start_writing
        )

    logger.info("Finished after %s seconds", time.time() - start)
