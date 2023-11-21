import dataclasses

import xarray as xr


@dataclasses.dataclass(frozen=True)
class Turbine:
    """Variables and meta data of wind turbine data."""

    production: str = "production"
    power_rating: str = "power rating"
    name: str = "wind plant"

    def read_power_rating(self, data: xr.Dataset) -> float:
        """Read the power rating from the meta data."""
        value = data.attrs[self.power_rating]
        if "kW" in value:
            return float(value.split()[0])
        return float(value)

    def get_turbine_name(self, data: xr.Dataset) -> str:
        """Read the turbine's name from the meta data."""
        if self.name in data.attrs:
            return data.attrs[self.name]
        return "nonamed"


@dataclasses.dataclass(frozen=True)
class Model:
    """Variables of model data."""

    u: str = "u"
    v: str = "v"
    wind_speed: str = "spd"
    wind_direction: str = "dir"
    sp: str = "sp"
    t: str = "t"
    r: str = "r"
    q: str = "q"
    z: str = "z"
    geopotential_height: str = "z_h"
