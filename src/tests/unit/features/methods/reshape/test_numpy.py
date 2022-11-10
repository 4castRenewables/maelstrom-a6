import numpy as np

import a6.features.methods.reshape.numpy as numpy


def test_reshape_spatio_temporal_data_array():
    data = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
        ],
    )

    expected = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ],
    )

    result = numpy.reshape_spatio_temporal_numpy_array(
        data=data, non_functional=True
    )

    np.testing.assert_equal(result, expected)
