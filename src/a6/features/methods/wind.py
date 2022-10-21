import logging

import a6.types as types
import numpy as np

logger = logging.getLogger(__name__)


def calculate_wind_speed(
    data: types.DataND,
    u: str = "u",
    v: str = "v",
) -> types.DataND:
    """Calculate the wind speed."""
    logger.debug(
        "Calculating wind speed using %s (zonal) and %s (meridional)", u, v
    )
    return np.sqrt(data[u] ** 2 + data[v] ** 2)


def calculate_wind_direction_angle(
    data: types.DataND,
    u: str = "u",
    v: str = "v",
) -> types.DataND:
    """Calculate the latitudinal wind direction angle."""
    logger.debug(
        "Calculating wind direction using %s (zonal) and %s (meridional)", u, v
    )
    angle = _calculate_angle_to_equator_in_deg(
        opposite=data[u], adjacent=data[v]
    )
    return 90.0 - angle


def _calculate_angle_to_equator_in_deg(
    opposite: types.DataND, adjacent: types.DataND
) -> types.DataND:
    angle_in_rad = np.arctan(opposite / adjacent)
    return _rad_to_deg(angle_in_rad)


def _rad_to_deg(angle: types.DataND) -> types.DataND:
    return angle * 360.0 / (2 * np.pi)
