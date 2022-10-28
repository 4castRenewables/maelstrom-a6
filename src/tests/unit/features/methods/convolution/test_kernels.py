import a6.features.methods.convolution.kernels as kernels
import numpy as np
import pytest


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
def test_create_average_kernel(size, expected):
    result = kernels.create_average_kernel(size)

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
    result = kernels.create_gaussian_kernel(size=size, sigma=sigma)

    np.testing.assert_almost_equal(result, expected, decimal=5)
