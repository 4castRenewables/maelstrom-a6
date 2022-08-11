import lifetimes.utils.standardization as standardization
import numpy as np


def test_standarize():
    data = np.array([1, 2, 3], dtype=float)
    mean = (1 + 2 + 3) / 3
    std = np.sqrt(((1 - 2) ** 2 * 1 + (2 - 2) ** 2 * 1 + (3 - 2) ** 2 * 1) / 3)
    expected = np.array([(1 - mean), (2 - mean), (3 - mean)]) / std

    result = standardization.standardize(data)

    np.testing.assert_equal(result, expected)
