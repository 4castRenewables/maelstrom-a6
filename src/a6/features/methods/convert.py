import logging
import pathlib
from collections.abc import Sequence

import cv2
import numpy as np
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.utils as utils

Variables = Sequence[str]
MinMaxValues = dict[str, tuple[float, float]]

logger = logging.getLogger(__name__)


@utils.log_consumption
def convert_fields_to_grayscale_images(
    data: xr.Dataset,
    variables: Variables,
    path: pathlib.Path = pathlib.Path("."),
    filename_prefix: str = "",
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
):
    """Convert three given fields to grayscale `.TIF` images.

    Parameters
    ----------
    data : xr.Dataset
        Data to be converted.
    variables : sequence[str, str, str]
        Variables whose fields to convert.
    path : pathlib.Path, default="."
        Path where to save the images.
    filename_prefix : str, default=""
        Filename prefix for the images.
        An expression stating the time step, level, and the physical fields
        processed will be appended, i.e.
        `<date>_level_<level>_<variable_1>_<variable_2>_<variable_3>`.
    coordinates : a6.datasets.coordinates.Coordinates
        Coordinates of the data.

    Notes
    -----
    Converts each given physical field into grayscale images. Here, each
    time step will result in an individual image with 3 channels consisting
    of the three given quantities (variables).

    """
    if len(variables) != 3:
        raise ValueError(
            "Exactly 3 fields required for converting to grayscale "
            f"(corresponding to RGB), got {variables!r}"
        )

    levels = data[coordinates.level].values
    try:
        level = int(levels)
    except TypeError:
        raise ValueError(
            "Can only convert single-level data to greyscale image, "
            f"given data has {levels.shape[0]} levels"
        )

    min_max_values = _get_min_max_values(data, variables=variables)

    logger.info("Starting conversion of images to .tif")

    for step in data[coordinates.time]:
        logger.debug("Converting time step %s", step.values)
        channels = _convert_fields_to_channels(
            data.sel(time=step),
            min_max_values=min_max_values,
        )
        _save_channels_to_image_file(
            channels,
            path=path,
            filename_prefix=filename_prefix,
            variables=variables,
            date=step,
            level=level,
        )


@utils.log_consumption
def _get_min_max_values(data: xr.Dataset, variables: Variables) -> MinMaxValues:
    logger.debug("Getting min and max values for data variables %s", variables)
    return {
        var: (data[var].min().compute(), data[var].max().compute())
        for var in variables
    }


@utils.log_consumption
def _convert_fields_to_channels(
    data: xr.Dataset, min_max_values: MinMaxValues
) -> np.ndarray:
    channels = [
        _convert_to_grayscale(data[var], min_=min_, max_=max_)
        for var, (min_, max_) in min_max_values.items()
    ]
    return np.array(channels, dtype=np.uint8).reshape(
        (*channels[0].shape, 3), order="F"
    )


@utils.log_consumption
def _convert_to_grayscale(d, min_, max_):
    return np.rint((d - min_) / max_ * 255)


@utils.log_consumption
def _save_channels_to_image_file(
    channels: np.ndarray,
    path: pathlib.Path,
    filename_prefix: str,
    variables: Variables,
    date: xr.DataArray,
    level: int,
) -> None:
    name = _create_file_name(
        path,
        filename_prefix=filename_prefix,
        variables=variables,
        date=date,
        level=level,
    )
    if not cv2.imwrite(name.as_posix(), channels):
        raise RuntimeError(f"Could not save image {name}")


def _create_file_name(
    path: pathlib.Path,
    filename_prefix: str,
    variables: Variables,
    date: xr.DataArray,
    level: int,
) -> pathlib.Path:
    variables_str = "_".join(variables)
    date_str = np.datetime_as_string(date.values, unit="m")

    if filename_prefix and filename_prefix[-1] != "_":
        filename_prefix = f"{filename_prefix}_"

    return pathlib.Path(
        path / f"{filename_prefix}{date_str}_level_{level}_{variables_str}.tif"
    )
