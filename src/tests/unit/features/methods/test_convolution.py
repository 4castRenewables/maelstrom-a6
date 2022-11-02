import a6.features.methods.convolution as convolution
import numpy as np
import pytest


@pytest.mark.parametrize(
    ("data", "kernel", "kwargs", "expected"),
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
            {},
            [
                [12.0 / 9.0, 12.0 / 9.0],
                [15.0 / 9.0, 15.0 / 9.0],
            ],
        ),
        (
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            "mean",
            {"size": 3},
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
            {},
            [
                [4.0 / 3.0, 4.0 / 3.0],
                [6.0 / 3.0, 6.0 / 3.0],
                [9.0 / 3.0, 9.0 / 3.0],
                [11.0 / 3.0, 11.0 / 3.0],
            ],
        ),
    ],
)
def test_apply_kernel(data, kernel, kwargs, expected):
    if isinstance(kernel, list):
        kernel = np.array(kernel)
    result = convolution.apply_kernel(np.array(data), kernel=kernel, **kwargs)

    np.testing.assert_equal(result, np.array(expected))


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (
            3,
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
        ),
    ],
)
def test_create_mean_kernel(size, expected):
    result = convolution.create_mean_kernel(size)

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("size", "sigma", "expected"),
    [
        (
            3,
            1,
            np.array(
                [
                    [0.36787944, 0.60653066, 0.36787944],
                    [0.60653066, 1.0, 0.60653066],
                    [0.36787944, 0.60653066, 0.36787944],
                ],
            ),
        ),
        (
            5,
            2,
            np.array(
                [
                    [
                        0.36787944,
                        0.53526143,
                        0.60653066,
                        0.53526143,
                        0.36787944,
                    ],
                    [0.53526143, 0.77880078, 0.8824969, 0.77880078, 0.53526143],
                    [0.60653066, 0.8824969, 1.0, 0.8824969, 0.60653066],
                    [0.53526143, 0.77880078, 0.8824969, 0.77880078, 0.53526143],
                    [
                        0.36787944,
                        0.53526143,
                        0.60653066,
                        0.53526143,
                        0.36787944,
                    ],
                ],
            ),
        ),
    ],
)
def test_create_gaussian_kernel(size, sigma, expected):
    result = convolution.create_gaussian_kernel(size=size, sigma=sigma)

    np.testing.assert_almost_equal(result, expected, decimal=5)
