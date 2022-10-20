import datetime

import a6.training.groups as groups
import xarray as xr


def test_get_group_labels_for_each_date():
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

    expected = [1, 1, 2]

    result = groups.get_group_labels_for_each_date(da)

    assert result == expected
