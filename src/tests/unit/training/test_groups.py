import datetime

import a6.training.groups as _groups
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def groups() -> _groups.Groups:
    da = xr.DataArray(
        [1, 2, 3],
        coords={
            "time": [
                datetime.datetime(2000, 1, 1, 1),
                datetime.datetime(2000, 1, 1, 2),
                datetime.datetime(2000, 1, 2, 1),
            ]
        },
    )

    return _groups.Groups(da)


class TestGroups:
    def test_labels(self, groups):
        expected = [1, 1, 2]

        result = groups.labels

        assert result == expected

    def test_evaluate_cross_validation(self, groups):
        cv = {
            "split0_test_mae": np.array([1]),
            "split0_test_nmae": np.array([2]),
            "split0_test_nrmse": np.array([3]),
            "split0_test_rmse": np.array([4]),
            "split1_test_mae": np.array([5]),
            "split1_test_nmae": np.array([6]),
            "split1_test_nrmse": np.array([7]),
            "split1_test_rmse": np.array([8]),
            "std_fit_time": np.array([0.00137843]),
            "std_score_time": np.array([0.00044371]),
            "std_test_mae": np.array([46.9218766]),
            "std_test_nmae": np.array([0.05520221]),
            "std_test_nrmse": np.array([0.05520221]),
            "std_test_rmse": np.array([46.9218766]),
        }

        expected = xr.Dataset(
            data_vars={
                "mae": (["time"], [1, 5]),
                "nmae": (["time"], [2, 6]),
                "nrmse": (["time"], [3, 7]),
                "rmse": (["time"], [4, 8]),
            },
            coords={
                "time": [
                    datetime.datetime(2000, 1, 1),
                    datetime.datetime(2000, 1, 2),
                ]
            },
        )

        result = groups.evaluate_cross_validation(
            cv, scores=["mae", "nmae", "rmse", "nrmse"]
        )

        xr.testing.assert_equal(result, expected)
