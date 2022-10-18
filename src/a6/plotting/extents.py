import typing as t


class Extent:
    longitude: slice
    latitude: slice

    @classmethod
    def to_dict(cls) -> t.Dict:
        return {
            "longitude": cls.longitude,
            "latitude": cls.latitude,
        }


class Germany(Extent):
    longitude: slice = slice(5.5, 15.3)
    latitude: slice = slice(55, 47)


class Extents:
    Germany: Germany = Germany
