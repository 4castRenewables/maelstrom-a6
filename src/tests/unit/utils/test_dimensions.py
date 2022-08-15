import lifetimes.utils.dimensions as _dimensions
import pytest


@pytest.fixture(scope="session")
def da_dimensions(da):
    return _dimensions.Dimensions.from_xarray(da, time_dimension="time")


@pytest.fixture(scope="session")
def ds_dimensions(ds):
    return _dimensions.Dimensions.from_xarray(ds, time_dimension="time")


@pytest.fixture(scope="session")
def ds_dimensions2(ds2):
    return _dimensions.Dimensions.from_xarray(ds2, time_dimension="time")


class TestDimensions:
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
            ("da_dimensions", (5, 10, 10)),
            ("ds_dimensions", (5, 10, 10)),
        ],
    )
    def test_to_tuple(self, request, dimensions, expected):
        dimensions = request.getfixturevalue(dimensions)

        result = dimensions.to_tuple()

        assert result == expected
