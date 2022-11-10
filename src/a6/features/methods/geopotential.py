import scipy.constants as constants

import a6.datasets.variables as _variables
import a6.types as types
import a6.utils as utils


@utils.make_functional
def calculate_geopotential_height(
    data: types.DataND,
    variables: _variables.Model = _variables.Model(),
    scaling: float = 10.0,
) -> types.DataND:
    """Calculate the geopotential height from the geopotential.

    Parameter
    ---------
    data : xr.Dataset
        Data containing the geopotential.
    variables : a6.datasets.variables.Model, optional
        Name of the variables.
    scaling : float, default=10.0
        Parameter used for scaling the data.
        E.g. the ECMWF IFS HRES geopotential is given in decameters.

    """
    data[variables.geopotential_height] = (
        data[variables.z] / constants.g / scaling
    )
    return data
