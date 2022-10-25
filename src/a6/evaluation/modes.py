import datetime

import a6.modes.methods.appearances as _appearances
import a6.utils as utils
import numpy as np
import xarray as xr

ScoresPerMode = dict[int, dict[str, np.ndarray]]


def evaluate_scores_per_mode(
    modes: _appearances.Modes,
    scores: list[xr.DataArray],
    time_coordinate: str = "time",
) -> ScoresPerMode:
    """Evaluate the scores that appeared at each mode."""

    return {
        mode.label: _get_score_values_for_mode(
            mode=mode,
            scores=scores,
            time_coordinate=time_coordinate,
        )
        for mode in modes
    }


def _get_score_values_for_mode(
    mode: _appearances.Mode, scores: list[xr.DataArray], time_coordinate: str
) -> dict[str, np.ndarray]:
    dates = _get_dates_from_mode(mode, time_coordinate=time_coordinate)
    result = {var: [] for var in list(scores[0].data_vars)}
    for score in scores:
        intersection = utils.get_time_step_intersection(
            dates, score, time_coordinate=time_coordinate
        )
        for var in score.data_vars:
            result[var].extend(
                _get_scores_for_variable(
                    score=score,
                    variable=var,
                    dates=intersection,
                    time_coordinate=time_coordinate,
                )
            )
    return _convert_to_numpy(result)


def _get_dates_from_mode(
    mode: _appearances.Mode, time_coordinate: str
) -> xr.DataArray:
    # Dates are required as an xarray.DataArray in order to calculate
    # intersecting time steps.
    dates = list(mode.get_dates())
    return xr.DataArray(dates, coords={time_coordinate: dates})


def _get_scores_for_variable(
    score: xr.DataArray,
    variable: str,
    dates: list[datetime.datetime],
    time_coordinate: str,
) -> list:
    values = score[variable].sel({time_coordinate: dates})
    return values.values.tolist()


def _convert_to_numpy(scores: dict[str, list]) -> dict[str, np.ndarray]:
    return {score: np.array(values) for score, values in scores.items()}
