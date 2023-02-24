import logging
import pathlib

import a6

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

level = 500
coordinates = a6.datasets.coordinates.Coordinates()
variables = a6.datasets.variables.Model()
path = pathlib.Path("/home/fabian/Documents/MAELSTROM/data/pca/")

preprocessing = a6.datasets.methods.select.select_dwd_area(
    coordinates=coordinates
) >> a6.datasets.methods.select.select_levels_and_calculate_daily_mean(
    levels=level
)

logger.info("Reading data")

data = a6.datasets.EcmwfIfsHres(
    path=path,
    pattern="pl_*.nc",
    slice_time_dimension=True,
    preprocessing=preprocessing,
    postprocessing=a6.features.methods.averaging.calculate_daily_mean(
        coordinates=coordinates
    ),
    parallel_loading=False,
).to_xarray()

processed = (
    a6.features.methods.weighting.weight_by_latitudes(
        latitudes=coordinates.latitude,
        use_sqrt=True,
    )
    >> a6.features.methods.geopotential.calculate_geopotential_height(
        variables=variables,
    )
    >> a6.features.methods.wind.calculate_wind_speed(variables=variables)
    >> a6.features.methods.variables.drop_variables(
        names=[variables.z, variables.u, variables.v]
    )
).apply_to(data)

processed = processed.isel(
    {coordinates.longitude: slice(128), coordinates.latitude: slice(128)}
)

variables_selection = [
    variables.t,
    variables.geopotential_height,
    variables.wind_speed,
]

a6.features.methods.convert.convert_fields_to_grayscale_images(
    processed,
    variables=variables_selection,
    path=path,
    filename_prefix="pl_",
)
