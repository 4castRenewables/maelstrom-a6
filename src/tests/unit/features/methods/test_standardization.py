import numpy as np
import xarray as xr

import a6.features.methods.standardization as standardization


def test_standarize():
    data = np.array([1, 2, 3], dtype=float)
    mean = (1 + 2 + 3) / 3
    std = np.sqrt(((1 - 2) ** 2 * 1 + (2 - 2) ** 2 * 1 + (3 - 2) ** 2 * 1) / 3)
    expected = np.array([(1 - mean), (2 - mean), (3 - mean)]) / std

    result = standardization.normalize(data, non_functional=True)

    np.testing.assert_equal(result, expected)


def test_standarize_features():
    data = xr.DataArray(
        [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
        ]
    )
    mean_1 = (1 + 2 + 3) / 3
    mean_2 = (4 + 5 + 6) / 3
    std_1 = np.sqrt(
        ((1 - mean_1) ** 2 + (2 - mean_1) ** 2 + (3 - mean_1) ** 2) / 3
    )
    std_2 = np.sqrt(
        ((4 - mean_2) ** 2 + (5 - mean_2) ** 2 + (6 - mean_2) ** 2) / 3
    )
    expected_1 = np.array([(1 - mean_1), (2 - mean_1), (3 - mean_1)]) / std_1
    expected_2 = np.array([(4 - mean_2), (5 - mean_2), (6 - mean_2)]) / std_2
    expected = xr.DataArray(list(zip(expected_1, expected_2)))

    result = standardization.normalize_features(data, non_functional=True)

    xr.testing.assert_equal(result, expected)
