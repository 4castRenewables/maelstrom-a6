import scipy.constants as constants
import xarray as xr


def calculate_geopotential_height(
    data: xr.Dataset, scaling: float = 10.0
) -> xr.Dataset:
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
