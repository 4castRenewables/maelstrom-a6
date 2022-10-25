import datetime
import pathlib

import a6.evaluation.modes as _modes
import a6.modes.methods.appearances as _appearances
import a6.utils as utils
import numpy as np
import xarray as xr

_FILE_DIR = pathlib.Path(__file__).parent


def test_evaluate_scores_per_mode(mode_appearances):
    scores_1 = xr.Dataset(
        data_vars={
            "nmae": (["time"], [1, 1]),
            "nrmse": (["time"], [2, 2]),
        },
        coords={
            "time": [
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 2),
            ]
        },
    )
    scores_2 = xr.Dataset(
        data_vars={
            "nmae": (["time"], [3, 3]),
            "nrmse": (["time"], [4, 4]),
        },
        coords={
            "time": [
                datetime.datetime(2000, 1, 3),
                datetime.datetime(2000, 1, 4),
            ]
        },
    )
    scores = [scores_1, scores_2]

    expected = {
        0: {
            "nmae": np.array([1, 1, 3]),
            "nrmse": np.array([2, 2, 4]),
        },
        1: {
            "nmae": np.array([3]),
            "nrmse": np.array([4]),
        },
    }

    result = _modes.evaluate_scores_per_mode(
        modes=mode_appearances, scores=scores
    )

    np.testing.assert_equal(result, expected)


def test_evaluate_scores_per_mode_with_real_data():
    data_dir = _FILE_DIR / "../../../../data"

    scores = xr.open_dataset(data_dir / "scores.nc")

    gwl = xr.open_dataset(data_dir / "gwl.nc")
    modes = _appearances.determine_lifetimes_of_modes(gwl["GWL"])

    result = _modes.evaluate_scores_per_mode(modes, scores=[scores])

    for step in scores["time"]:
        date = utils.numpy_datetime64_to_datetime(step.values)
        appearence = modes.get_appearance(
            utils.numpy_datetime64_to_datetime(date)
        )
        res = result[appearence.label]
        for score in scores.data_vars:
            r = res[score]
            exp = scores[score].sel(time=date).values
            np.testing.assert_equal(r, exp)
