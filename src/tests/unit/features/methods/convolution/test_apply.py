import a6.features.methods.convolution.apply as apply
import numpy as np
import pytest


@pytest.mark.parametrize(
    ("data", "kernel", "expected"),
    [
        (
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [12.0 / 9.0, 12.0 / 9.0],
                [15.0 / 9.0, 15.0 / 9.0],
            ],
        ),
        (
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ],
            [
                [1.0],
                [1.0],
                [1.0],
            ],
            [
                [4.0 / 3.0, 4.0 / 3.0],
                [6.0 / 3.0, 6.0 / 3.0],
                [9.0 / 3.0, 9.0 / 3.0],
                [11.0 / 3.0, 11.0 / 3.0],
            ],
        ),
    ],
)
def test_apply_kernel(data, kernel, expected):
    result = apply.apply_kernel(np.array(data), kernel=np.array(kernel))

    np.testing.assert_equal(result, np.array(expected))


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
    result = apply.apply_pooling(data, size=size, mode=mode)

    np.testing.assert_equal(result, expected)
