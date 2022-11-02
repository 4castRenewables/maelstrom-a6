import a6.features.methods.pooling as pooling
import numpy as np
import pytest


@pytest.mark.parametrize(
    ("data", "size", "mode", "expected"),
    [
        (
            np.array(
                [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 4, 4],
                    [3, 3, 4, 4],
                ]
            ),
            2,
            "mean",
            np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
        ),
        (
            np.array(
                [
                    [0, 1, 0, 2],
                    [1, 2, 2, 3],
                    [0, 3, 0, 4],
                    [3, 4, 4, 5],
                ]
            ),
            2,
            "median",
            np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 2, 1, 2],
                    [3, 4, 3, 4],
                    [1, 2, 1, 2],
                    [3, 4, 3, 4],
                ]
            ),
            2,
            "max",
            np.array(
                [
                    [4, 4],
                    [4, 4],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 2, 1, 2],
                    [3, 4, 3, 4],
                    [1, 2, 1, 2],
                    [3, 4, 3, 4],
                ]
            ),
            2,
            "min",
            np.array(
                [
                    [1, 1],
                    [1, 1],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 1, 2],
                    [1, 1, 2],
                    [3, 3, 4],
                ]
            ),
            2,
            "mean",
            np.array(
                [
                    [4.0 / 4.0, 8.0 / 4.0],
                    [10.0 / 4.0, 10.0 / 4.0],
                ]
            ),
        ),
    ],
)
def test_apply_pooling(data, size, mode, expected):
    result = pooling.apply_pooling(data, size=size, mode=mode)

    np.testing.assert_equal(result, expected)
