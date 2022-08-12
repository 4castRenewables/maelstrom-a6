import lifetimes.utils.dimensions as shape
import pytest


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("da", (5, 10, 10)),
        ("ds", (5, 10, 10, 1)),
    ],
)
def test_get_xarray_data_dimensions(request, data, expected):
    data = request.getfixturevalue(data)

    result = shape.get_xarray_data_dimensions(data)

    assert result == expected
