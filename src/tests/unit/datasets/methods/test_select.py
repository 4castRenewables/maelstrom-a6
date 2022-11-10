import numpy as np
import pytest

import a6.datasets.methods.select as select


@pytest.mark.parametrize("levels", [500, [500, 1000]])
def test_select_levels(pl_ds, levels):
    result = select.select_levels(pl_ds, levels=levels, non_functional=True)

    assert result["level"].values.tolist() == levels


def test_select_levels_and_calculate_daily_mean(pl_ds):
    expected = np.array(
        [
            [
                [245.4356, 245.53772],
                [245.34428, 245.38219],
                [245.47972, 245.4964],
            ],
            [
                [235.93938, 235.96883],
                [235.544, 235.57054],
                [235.26147, 235.30586],
            ],
            [
                [233.545, 233.50867],
                [232.33002, 232.33957],
                [232.06425, 232.16368],
            ],
        ],
        dtype=np.float32,
    )

    result = select.select_levels_and_calculate_daily_mean(
        pl_ds, levels=500, non_functional=True
    )

    assert result["level"].values == 500

    np.testing.assert_allclose(result["t"].values, expected)
