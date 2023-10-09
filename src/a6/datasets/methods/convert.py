import json
import logging
import pathlib
from collections.abc import Callable
from collections.abc import Sequence

import cv2
import numpy as np
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.datasets.methods.normalization as normalization
import a6.utils as utils

Variables = Sequence[str]
MinMaxValues = dict[str, tuple[float, float]]
FileNameFactory = Callable[[xr.DataArray], pathlib.Path]

logger = logging.getLogger(__name__)


class _FileNameCreator:
    def __init__(
        self,
        path: pathlib.Path,
        filename_prefix: str,
        variables: Variables,
        level: int,
    ):
        self._path = path
        self._filename_prefix = (
            f"{filename_prefix}_"
            if filename_prefix and filename_prefix[-1] != "_"
            else filename_prefix
        )
        self._variables = variables
        self._variables_str = "_".join(variables)
        self._level = level

    def create(self, date: xr.DataArray) -> pathlib.Path:
        date_str = np.datetime_as_string(date.values, unit="m")
        return pathlib.Path(
            self._path
            / (
                f"{self._filename_prefix}{date_str}_"
                f"level_{self._level}_"
                f"{self._variables_str}"
                ".tif"
            )
        )


@utils.log_consumption
def convert_fields_to_grayscale_images(
    data: xr.Dataset,
    variables: Variables,
    path: pathlib.Path = pathlib.Path("../../features/methods"),
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

    min_max_values = normalization.get_min_max_values(data, variables=variables)

    with open(path / "min_max_values.json", "w") as f:
        f.write(json.dumps(min_max_values, sort_keys=True, indent=2))

    logger.info("Starting conversion of images to .tif")
    filename_creator = _FileNameCreator(
        path=path,
        filename_prefix=filename_prefix,
        variables=variables,
        level=level,
    )
    steps = (step for step in data[coordinates.time])
    kwargs = dict(
        data=data,
        min_max_values=min_max_values,
        filename_creator=filename_creator,
    )
    utils.parallelize(
        function=_convert_time_step_and_save_to_file,
        args_zipped=steps,
        single_arg=True,
        kwargs_as_dict=kwargs,
    )


def _convert_time_step_and_save_to_file(
    time_step: xr.DataArray,
    data: xr.Dataset,
    min_max_values: list[normalization.VariableMinMax],
    filename_creator: _FileNameCreator,
) -> None:
    logger.debug("Converting time step %s", time_step.values)
    channels = _convert_fields_to_channels(
        data.sel(time=time_step),
        min_max_values=min_max_values,
    )
    _save_channels_to_image_file(
        channels,
        date=time_step,
        filename_creator=filename_creator,
    )


@utils.log_consumption
def _convert_fields_to_channels(
    data: xr.Dataset, min_max_values: list[normalization.VariableMinMax]
) -> np.ndarray:
    channels = [
        _convert_to_grayscale(
            data[variable.name], min_=variable.min, max_=variable.max
        )
        for variable in min_max_values
    ]
    return np.array(channels, dtype=np.uint8).reshape(
        (*channels[0].shape, 3), order="F"
    )


@utils.log_consumption
def _convert_to_grayscale(
    data: xr.Dataset, min_max: normalization.VariableMinMax
) -> xr.DataArray:
    return np.rint(
        normalization.min_max_scale_variable(data, min_max=min_max) * 255
    )


@utils.log_consumption
def _save_channels_to_image_file(
    channels: np.ndarray,
    filename_creator: _FileNameCreator,
    date: xr.DataArray,
) -> None:
    name = filename_creator.create(date)
    if not cv2.imwrite(name.as_posix(), channels):
        raise RuntimeError(f"Could not save image {name}")
