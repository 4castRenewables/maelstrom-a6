import dataclasses


@dataclasses.dataclass(frozen=True)
class Coordinates:
    """Name of the coordinates of a dataset."""

    time: str = "time"
    latitude: str = "latitude"
    longitude: str = "longitude"
    level: str = "level"
