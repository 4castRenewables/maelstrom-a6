import a6.evaluation.residuals as residuals
import numpy as np
import pytest
import xarray as xr


@pytest.fixture()
def left() -> xr.DataArray:
    return xr.DataArray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )


@pytest.fixture()
def right() -> xr.DataArray:
    return xr.DataArray(
        [
            [2.0, 4.0],
            [6.0, 8.0],
        ]
    )


@pytest.fixture()
def expected_ssr() -> float:
    return sum(
        [
            (1.0 - 2.0) ** 2,
            (2.0 - 4.0) ** 2,
            (3.0 - 6.0) ** 2,
            (4.0 - 8.0) ** 2,
        ]
    )


def test_calculate_ssr(left, right, expected_ssr):
    result = residuals.calculate_ssr(left, right)

    assert result == expected_ssr


def test_calculate_normalized_root_ssr(left, right, expected_ssr):
    expected = np.sqrt(expected_ssr) / 4.0

    result = residuals.calculate_normalized_root_ssr(left, right)

    assert result == expected
