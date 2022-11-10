import datetime

import numpy as np
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.modes.methods.appearances as _appearances
import a6.utils as utils

ScoresPerMode = dict[int, dict[str, np.ndarray]]


def evaluate_scores_per_mode(
    modes: _appearances.Modes,
    scores: list[xr.DataArray],
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> ScoresPerMode:
    """Evaluate the scores that appeared at each mode."""

    return {
        mode.label: _get_score_values_for_mode(
            mode=mode,
            scores=scores,
            coordinates=coordinates,
        )
        for mode in modes
    }


def _get_score_values_for_mode(
    mode: _appearances.Mode,
    scores: list[xr.DataArray],
    coordinates: _coordinates.Coordinates,
) -> dict[str, np.ndarray]:
    dates = _get_dates_from_mode(mode, coordinates=coordinates)
    result = {var: [] for var in list(scores[0].data_vars)}
    for score in scores:
        intersection = utils.get_time_step_intersection(
            dates,
            score,
            coordinates=coordinates,
        )
        for var in score.data_vars:
            result[var].extend(
                _get_scores_for_variable(
                    score=score,
                    variable=var,
                    dates=intersection,
                    coordinates=coordinates,
                )
            )
    return _convert_to_numpy(result)


def _get_dates_from_mode(
    mode: _appearances.Mode, coordinates: _coordinates.Coordinates
) -> xr.DataArray:
    # Dates are required as an xarray.DataArray in order to calculate
    # intersecting time steps.
    dates = list(mode.get_dates())
    return xr.DataArray(dates, coords={coordinates.time: dates})


def _get_scores_for_variable(
    score: xr.DataArray,
    variable: str,
    dates: list[datetime.datetime],
    coordinates: _coordinates.Coordinates,
) -> list:
    values = score[variable].sel({coordinates.time: dates})
    return values.values.tolist()


def _convert_to_numpy(scores: dict[str, list]) -> dict[str, np.ndarray]:
    return {score: np.array(values) for score, values in scores.items()}
