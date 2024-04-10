import argparse
import logging
import os
import pathlib

import xarray as xr

import a6
import a6.datasets.coordinates as _coordinates
import a6.datasets.variables as _variables
import a6.utils as utils

WORKER_ID = (
    int(os.getenv("SLURM_PROCID")) if "SLURM_PROCID" in os.environ else None
)

utils.logging.create_logger(
    global_rank=WORKER_ID,
    local_rank=WORKER_ID,
    verbose=True,
)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pressure-level-data",
    type=pathlib.Path,
)
parser.add_argument(
    "--model-level-data",
    type=pathlib.Path,
)
parser.add_argument(
    "--surface-level-data",
    type=pathlib.Path,
)
parser.add_argument(
    "--turbine-data-dir",
    type=pathlib.Path,
)
parser.add_argument(
    "--output-dir",
    type=pathlib.Path,
)


def create_turbine_model_features(
    raw_args: list[str] | None = None,
) -> dict[pathlib.Path, xr.Dataset]:
    """Create the input data for simulation of the forecasts."""
    args = parser.parse_args(raw_args)

    turbine_files = utils.paths.list_files(
        args.turbine_data_dir, pattern="**/*.nc", recursive=True
    )

    if WORKER_ID is not None and WORKER_ID >= len(turbine_files):
        logger.warning("Exiting: no file to process")
        return

    coordinates: _coordinates.Coordinates = _coordinates.Coordinates()
    model_variables: _variables.Model = _variables.Model()
    turbine_variables: _variables.Turbine = a6.datasets.variables.Turbine()

    ds_sfc = xr.open_dataset(args.surface_level_data)
    ds_ml = xr.open_dataset(args.model_level_data).sel({coordinates.level: 137})
    ds_pl = xr.open_dataset(args.pressure_level_data).sel(
        {coordinates.level: 1000}
    )

    outfiles = {}

    for i, turbine_path in enumerate(turbine_files):
        if WORKER_ID is not None and i != WORKER_ID:
            continue

        logger.info(
            "Processing turbine %i/%i (path=%s)",
            i,
            len(turbine_files),
            turbine_path,
        )

        turbine_name = turbine_path.name.replace(".nc", "")
        outfile_turbine: pathlib.Path = (
            args.output_dir / f"{turbine_name}/turbine.nc"
        )

        if not outfile_turbine.parent.exists():
            logger.info(
                "Creating directory for outfiles of turbine %s at %s",
                turbine_name,
                outfile_turbine.parent,
            )
            outfile_turbine.parent.mkdir(exist_ok=True, parents=True)

        outfile_pl: pathlib.Path = args.output_dir / f"{turbine_name}/pl.nc"
        outfile_ml: pathlib.Path = args.output_dir / f"{turbine_name}/ml.nc"
        outfile_sfc: pathlib.Path = args.output_dir / f"{turbine_name}/sfc.nc"

        turbine = xr.open_dataset(turbine_path)

        power_rating = turbine_variables.read_power_rating(turbine)
        logger.info("Extracted power rating %i", power_rating)

        logger.info("Preprocessing turbine data")
        try:
            turbine = (
                a6.datasets.methods.turbine.clean_production_data(
                    power_rating=power_rating,
                    variables=turbine_variables,
                )
                >> a6.datasets.methods.turbine.resample_to_hourly_resolution(
                    variables=turbine_variables,
                    coordinates=coordinates,
                )
                >> a6.datasets.methods.select.select_latitude_longitude(
                    latitude=0, longitude=0
                )
                >> a6.datasets.methods.select.select_intersecting_time_steps(
                    right=ds_ml,
                    coordinates=coordinates,
                )
                >> a6.datasets.methods.save.to_netcdf(path=outfile_turbine)
            ).apply_to(turbine)
        except ValueError:
            logger.exception(
                (
                    "Skipping: No intersecting time steps for "
                    "turbine %s and given data, deleting directory"
                ),
                turbine_name,
            )
            outfile_turbine.parent.rmdir()
            continue

        logger.info("Preprocessing surface level data")
        result_sfc = (
            a6.datasets.methods.turbine.get_closest_grid_point(
                turbine=turbine,
                coordinates=coordinates,
            )
            >> a6.datasets.methods.select.select_intersecting_time_steps(
                right=turbine, coordinates=coordinates
            )
            >> a6.datasets.methods.select.select_variables(
                variables=model_variables.sp
            )
            >> a6.datasets.methods.save.to_netcdf(path=outfile_sfc)
        ).apply_to(ds_sfc)

        logger.info("Preprocessing model level data")
        result_ml = (
            a6.datasets.methods.turbine.get_closest_grid_point(
                turbine=turbine,
                coordinates=coordinates,
            )
            >> a6.datasets.methods.select.select_intersecting_time_steps(
                right=turbine, coordinates=coordinates
            )
            >> a6.datasets.methods.select.select_variables(
                variables=model_variables.t
            )
            >> a6.datasets.methods.save.to_netcdf(path=outfile_ml)
        ).apply_to(ds_ml)

        logger.info("Preprocessing pressure level data")
        result_pl = (
            a6.datasets.methods.turbine.get_closest_grid_point(
                turbine=turbine,
                coordinates=coordinates,
            )
            >> a6.datasets.methods.select.select_intersecting_time_steps(
                right=turbine, coordinates=coordinates
            )
            >> a6.features.methods.wind.calculate_wind_speed(
                variables=model_variables
            )
            >> a6.features.methods.wind.calculate_wind_direction_angle(
                variables=model_variables
            )
            >> a6.features.methods.time.calculate_fraction_of_day(
                coordinates=coordinates
            )
            >> a6.features.methods.time.calculate_fraction_of_year(
                coordinates=coordinates
            )
            >> a6.datasets.methods.select.select_variables(
                variables=[
                    model_variables.wind_speed,
                    model_variables.wind_direction,
                    model_variables.r,
                    "fraction_of_year",
                    "fraction_of_day",
                ]
            )
            >> a6.datasets.methods.save.to_netcdf(path=outfile_pl)
        ).apply_to(ds_pl)

        outfiles = outfiles | {
            outfile_turbine: turbine,
            outfile_ml: result_ml,
            outfile_pl: result_pl,
            outfile_sfc: result_sfc,
        }

    return outfiles
