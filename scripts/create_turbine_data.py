import logging
import pathlib
import time

import xarray as xr

import a6
import a6.datasets.coordinates as _coordinates
import a6.datasets.variables as _variables
import a6.utils as utils

utils.log_to_stdout()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info("initiated")

output_dir = pathlib.Path(
    "/p/home/jusers/emmerich1/juwels/data/production-processed"
)

turbine_data_dir = "/p/home/jusers/emmerich1/juwels/data/production"
surface_level_data = (
    "/p/project1/deepacf/emmerich1/data/ecmwf_ifs/sfc_2017_2020.nc"
)
model_level_data = "/p/project1/deepacf/emmerich1/data/ecmwf_ifs/ml_2017_2020.nc"
pressure_level_data = (
    "/p/project1/deepacf/emmerich1/data/ecmwf_ifs/pl_2017_2020.nc"
)


@utils.log_consumption
@utils.make_functional
def select_intersecting_time_steps(
    left: xr.Dataset,
    right: xr.Dataset,
    return_only_left: bool = True,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset | tuple[xr.Dataset, xr.Dataset]:
    """Select the overlapping time steps of the datasets."""
    intersection = utils.get_time_step_intersection(
        left=left,
        right=right,
        coordinates=coordinates,
    )

    if not intersection:
        logger.warning("No intersection found for %s", left)

    select = {coordinates.time: intersection}
    if return_only_left:
        return left.sel(select)
    return left.sel(select), right.sel(select)


@utils.log_consumption
@utils.make_functional
def to_netcdf(ds: xr.Dataset, *, path: pathlib.Path) -> xr.Dataset:
    """Save dataset to netcdf file."""
    logger.info("Saving dataset %s to disk at %s", ds, path)
    start = time.time()
    ds.to_netcdf(path)
    logger.info("Saving finished in %.2f seconds", time.time() - start)
    return ds


turbine_files = utils.paths.list_files(turbine_data_dir, pattern="**/*.nc")

coordinates: _coordinates.Coordinates = _coordinates.Coordinates()
model_variables: _variables.Model = _variables.Model()
turbine_variables: _variables.Turbine = a6.datasets.variables.Turbine()

ds_sfc = xr.open_dataset(surface_level_data)
ds_ml = xr.open_dataset(model_level_data).sel({coordinates.level: 137})
ds_pl = xr.open_dataset(pressure_level_data).sel({coordinates.level: 1000})

for i, turbine_path in enumerate(turbine_files):
    logger.info(
        "Processing turbine %i/%i (path=%s)",
        i,
        len(turbine_files),
        turbine_path,
    )

    turbine_name = turbine_path.name.replace(".nc", "")

    if i < 32:
        logger.info(
            "Skipping turbine %s (%i/%i): Already processed",
            turbine_name,
            i,
            len(turbine_files),
        )
        continue

    outfile_turbine: pathlib.Path = output_dir / f"{turbine_name}/turbine.nc"

    if not outfile_turbine.parent.exists():
        outfile_turbine.parent.mkdir(exist_ok=True, parents=True)

    outfile_pl: pathlib.Path = output_dir / f"{turbine_name}/pl.nc"
    outfile_ml: pathlib.Path = output_dir / f"{turbine_name}/ml.nc"
    outfile_sfc: pathlib.Path = output_dir / f"{turbine_name}/sfc.nc"

    turbine = xr.open_dataset(turbine_path)

    power_rating = turbine_variables.read_power_rating(turbine)
    logger.info("Extracted power rating %i", power_rating)

    logger.info("Preprocessing turbine data")
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
        >> select_intersecting_time_steps(
            right=ds_ml,
            coordinates=coordinates,
        )
        >> to_netcdf(path=outfile_turbine)
    ).apply_to(turbine)

    if turbine[coordinates.time].size == 0:
        logger.warning(
            (
                "Skipping: No intersecting time steps for turbine %s "
                "and given data"
            ),
            turbine_name,
        )
        continue

    logger.info("Preprocessing surface level data")
    (
        a6.datasets.methods.turbine.get_closest_grid_point(
            turbine=turbine,
            coordinates=coordinates,
        )
        >> select_intersecting_time_steps(
            right=turbine, coordinates=coordinates
        )
        >> a6.datasets.methods.select.select_variables(
            variables=model_variables.sp
        )
        >> to_netcdf(path=outfile_sfc)
    ).apply_to(ds_sfc)

    logger.info("Preprocessing model level data")
    (
        a6.datasets.methods.turbine.get_closest_grid_point(
            turbine=turbine,
            coordinates=coordinates,
        )
        >> select_intersecting_time_steps(
            right=turbine, coordinates=coordinates
        )
        >> a6.datasets.methods.select.select_variables(
            variables=model_variables.t
        )
        >> to_netcdf(path=outfile_ml)
    ).apply_to(ds_ml)

    logger.info("Preprocessing pressure level data")
    (
        a6.datasets.methods.turbine.get_closest_grid_point(
            turbine=turbine,
            coordinates=coordinates,
        )
        >> select_intersecting_time_steps(
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
        >> to_netcdf(path=outfile_pl)
    ).apply_to(ds_pl)
