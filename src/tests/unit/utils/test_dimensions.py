import lifetimes.utils.coordinates as _coordinates
import lifetimes.utils.dimensions as _dimensions
import pytest


@pytest.fixture(scope="session")
def da_dimensions(da, coordinates):
    return _dimensions.SpatioTemporalDimensions.from_xarray(
        da,
        coordinates=coordinates,
    )


@pytest.fixture(scope="session")
def ds_dimensions(ds, coordinates):
    return _dimensions.SpatioTemporalDimensions.from_xarray(
        ds,
        coordinates=coordinates,
    )


@pytest.fixture(scope="session")
def ds_dimensions2(ds2, coordinates):
    return _dimensions.SpatioTemporalDimensions.from_xarray(
        ds2, coordinates=coordinates
    )


@pytest.fixture(scope="session")
def ds_dimensions_in_real_order(pl_ds):
    return _dimensions.SpatioTemporalDimensions.from_xarray(
        pl_ds.sel(level=500), coordinates=_coordinates.CoordinateNames()
    )


class TestSpatioTemporalDimensions:
    @pytest.mark.parametrize(
        ("dimensions", "expected"),
        [
            ("da_dimensions", False),
            ("ds_dimensions", False),
            ("ds_dimensions2", True),
        ],
    )
    def test_is_multi_variable(self, request, dimensions, expected):
        dimensions = request.getfixturevalue(dimensions)

        result = dimensions.is_multi_variable

        assert result == expected

    @pytest.mark.parametrize(
        ("dimensions", "expected"),
        [
            ("da_dimensions", ("lat", "lon")),
            ("ds_dimensions", ("lat", "lon")),
            ("ds_dimensions2", ("lat", "lon")),
            ("ds_dimensions_in_real_order", ("latitude", "longitude")),
        ],
    )
    def test_spatial_dimension_names(self, request, dimensions, expected):
        dimensions = request.getfixturevalue(dimensions)

        result = dimensions.spatial_dimension_names

        assert result == expected

    @pytest.mark.parametrize(
        ("dimensions", "expected"),
        [
            ("da_dimensions", (5, 10, 10)),
            ("ds_dimensions", (5, 10, 10)),
            ("ds_dimensions2", (5, 10, 10)),
            ("ds_dimensions_in_real_order", (49, 3, 2)),
        ],
    )
    def test_shape(self, request, dimensions, expected):
        dimensions = request.getfixturevalue(dimensions)

        result = dimensions.shape()

        assert result == expected
