import numpy as np
from a6.utils.reshape import numpy_array


def test_reshape_spatio_temporal_xarray_data_array():
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

    result = numpy_array.reshape_spatio_temporal_numpy_array(data=data)

    np.testing.assert_equal(result, expected)
