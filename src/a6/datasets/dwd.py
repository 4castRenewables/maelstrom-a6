import logging
import pathlib

import a6.datasets as datasets
import a6.features as features

logger = logging.getLogger(__name__)


def get_dwd_era5_data(
    path: pathlib.Path,
    pattern: str,
    parallel_loading: bool,
    levels: list[int] | None = None,
    drop_variables: list[str] | None = None,
    select_dwd_area: bool = True,
):
    # If a data pattern is given, it is assumed that the
    # given data path is a folder with netCDF files.
    logger.warning("Assuming xarray.Dataset from netCDF files")
    coordinates = datasets.coordinates.Coordinates()
    variables = datasets.variables.Model()
    drop_variables = drop_variables or []

    preprocessing = (
        (
            datasets.methods.select.select_dwd_area(coordinates=coordinates)
            if select_dwd_area
            else datasets.methods.identity.identity()
        )
        >> features.methods.weighting.weight_by_latitudes(
            latitudes=coordinates.latitude,
            use_sqrt=True,
        )
        >> features.methods.geopotential.calculate_geopotential_height(
            variables=variables,
        )
        >> features.methods.variables.drop_variables(
            names=[variables.z] + drop_variables
        )
    )

    logger.info(
        "Reading data from netCDF files and converting to xarray.Dataset"
    )
    ds = datasets.Era5(
        path=path,
        pattern=pattern,
        preprocessing=preprocessing,
        parallel_loading=parallel_loading,
    ).to_xarray(levels=levels)

    return ds
