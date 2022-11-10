import logging

import numpy as np

import a6.datasets.variables as _variables
import a6.types as types
import a6.utils as utils

logger = logging.getLogger(__name__)


@utils.make_functional
def calculate_wind_speed(
    data: types.DataND,
    variables: _variables.Model = _variables.Model(),
) -> types.DataND:
    """Calculate the wind speed."""
    logger.debug("Calculating wind speed using %s", variables)
    data[variables.wind_speed] = np.sqrt(
        data[variables.u] ** 2 + data[variables.v] ** 2
    )
    return data


@utils.make_functional
def calculate_wind_direction_angle(
    data: types.DataND,
    variables: _variables.Model = _variables.Model(),
) -> types.DataND:
    """Calculate the latitudinal wind direction angle."""
    logger.debug("Calculating wind direction using %s", variables)
    angle = _calculate_angle_to_equator_in_deg(
        opposite=data[variables.u], adjacent=data[variables.v]
    )
    data[variables.wind_direction] = 90.0 - angle
    return data


def _calculate_angle_to_equator_in_deg(
    opposite: types.DataND, adjacent: types.DataND
) -> types.DataND:
    angle_in_rad = np.arctan(opposite / adjacent)
    return _rad_to_deg(angle_in_rad)


def _rad_to_deg(angle: types.DataND) -> types.DataND:
    return angle * 360.0 / (2 * np.pi)
