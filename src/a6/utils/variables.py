import dataclasses

import xarray as xr


@dataclasses.dataclass(frozen=True)
class Turbine:
    """Variables and meta data of wind turbine data."""

    production: str = "production"
    power_rating: str = "power rating"

    def read_power_rating(self, data: xr.Dataset) -> float:
        """Read the power rating from the meta data."""
        value = data.attrs[self.power_rating]
        if "kW" in value:
            return float(value.split()[0])
        return float(value)


@dataclasses.dataclass(frozen=True)
class Model:
    """Variables of model data."""

    u: str = "u"
    v: str = "v"
