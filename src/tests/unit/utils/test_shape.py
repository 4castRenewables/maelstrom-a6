import lifetimes.utils.shape as shape
import pytest


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("da", (5, 10, 10)),
        ("ds", (5, 10, 10, 1)),
    ],
)
def test_get_xarray_data_shape(request, data, expected):
    data = request.getfixturevalue(data)

    result = shape.get_xarray_data_shape(data)

    assert result == expected
