import a6.types as types
import scipy.constants as constants


def calculate_geopotential_height(
    data: types.DataND, scaling: float = 10.0
) -> types.DataND:
    """Calculate the geopotential height from the geopotential.

    Parameter
    ---------
    data : xr.Dataset
        Data containing the geopotential.
    scaling : float, default=10.0
        Parameter used for scaling the data.
        E.g. the ECMWF IFS HRES geopotential is given in decameters.


    """
    return data / constants.g / scaling
