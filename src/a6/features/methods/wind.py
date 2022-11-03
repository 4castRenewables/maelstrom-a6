import logging

import a6.datasets.variables as _variables
import a6.types as types
import numpy as np

logger = logging.getLogger(__name__)


def calculate_wind_speed(
    data: types.DataND,
    variables: _variables.Model = _variables.Model(),
) -> types.DataND:
    """Calculate the wind speed."""
    logger.debug("Calculating wind speed using %s", variables)
    return np.sqrt(data[variables.u] ** 2 + data[variables.v] ** 2)


def calculate_wind_direction_angle(
    data: types.DataND,
    variables: _variables.Model = _variables.Model(),
) -> types.DataND:
    """Calculate the latitudinal wind direction angle."""
    logger.debug("Calculating wind direction using %s", variables)
    angle = _calculate_angle_to_equator_in_deg(
        opposite=data[variables.u], adjacent=data[variables.v]
    )
    return 90.0 - angle


def _calculate_angle_to_equator_in_deg(
    opposite: types.DataND, adjacent: types.DataND
) -> types.DataND:
    angle_in_rad = np.arctan(opposite / adjacent)
    return _rad_to_deg(angle_in_rad)


def _rad_to_deg(angle: types.DataND) -> types.DataND:
    return angle * 360.0 / (2 * np.pi)
