import argparse
import dataclasses
import datetime
import itertools
import logging
import pathlib

import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.metrics
import sklearn.model_selection
import xarray as xr

import a6
import a6.datasets.coordinates as _coordinates
import a6.datasets.variables as _variables

a6.utils.logging.log_to_stdout(level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--turbine-data-dir",
    type=pathlib.Path,
)
parser.add_argument(
    "--preprocessed-data-dir",
    type=pathlib.Path,
)
parser.add_argument(
    "--results-dir",
    type=pathlib.Path,
)
parser.add_argument(
    "--parallel",
    type=bool,
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--testing",
    type=bool,
    action=argparse.BooleanOptionalAction,
    default=False,
)

ForecastErrors = dict[pathlib.Path, xr.Dataset]


def simulate_errors(
    raw_args: list[str] | None = None,
) -> ForecastErrors:
    """For a set of turbines, simulate forecasts and calculate the errors.

    The turbines are considered to be located in a certain directory
    as `.nc` files, where each file contains the production data
    of the respective turbine.

    """
    args = parser.parse_args(raw_args)

    turbine_files = a6.utils.paths.list_files(
        args.turbine_data_dir, pattern="**/*.nc", recursive=True
    )

    coordinates: _coordinates.Coordinates = _coordinates.Coordinates()
    turbine_variables: _variables.Turbine = a6.datasets.variables.Turbine()

    result: ForecastErrors = {}

    for i, turbine_path in enumerate(turbine_files):
        logger.info(
            "Processing turbine %i/%i (path=%s)",
            i,
            len(turbine_files),
            turbine_path,
        )

        turbine_name = turbine_path.name.replace(".nc", "")
        outfile: pathlib.Path = (
            args.results_dir / f"{turbine_name}_forecast_errors.nc"
        )

        if outfile.exists():
            logger.warning(
                "Skipping %s since outfile already exists at %s",
                turbine_path,
                outfile,
            )

        turbine_path: pathlib.Path = (
            args.preprocessed_data_dir / f"{turbine_name}/turbine.nc"
        )
        pl_path: pathlib.Path = (
            args.preprocessed_data_dir / f"{turbine_name}/pl.nc"
        )
        ml_path: pathlib.Path = (
            args.preprocessed_data_dir / f"{turbine_name}/ml.nc"
        )
        sfc_path: pathlib.Path = (
            args.preprocessed_data_dir / f"{turbine_name}/sfc.nc"
        )

        logger.info("Reading preprocessed data")
        turbine = xr.open_dataset(turbine_path)
        pl = xr.open_dataset(pl_path)
        ml = xr.open_dataset(ml_path)
        sfc = xr.open_dataset(sfc_path)

        power_rating = turbine_variables.read_power_rating(turbine)
        logger.info("Extracted power rating %i", power_rating)

        data = [sfc, ml] + [pl[var] for var in pl.data_vars]

        logger.info(
            "Preparing input data for variables %s", [d.name for d in data]
        )

        X = a6.features.methods.reshape.sklearn.transpose(*data)  # noqa: N806
        y = a6.features.methods.reshape.sklearn.transpose(
            turbine[turbine_variables.production]
        )

        (  # noqa: N806
            X_train,
            _,
            y_train,
            _,
        ) = sklearn.model_selection.train_test_split(X, y, train_size=0.8)

        logger.info(
            "Train dataset size is %i hours (~%i days)",
            y_train.size,
            y_train.size / 24,
        )

        logger.info("Fitting model with GridSearchCV")

        if args.testing:
            param_grid = {"learning_rate": [0.1]}
        else:
            param_grid = {
                "learning_rate": [0.03, 0.05, 0.07, 0.1],
                "l2_regularization": [0.0, 1.0, 3.0, 5.0, 7.0],
                "max_iter": [200, 300, 500],
                "max_depth": [15, 37, 63, 81],
                "min_samples_leaf": [23, 48, 101, 199],
            }
        n_jobs = -1 if args.parallel else int(a6.utils.get_cpu_count() / 2)

        gs = sklearn.model_selection.GridSearchCV(
            estimator=ensemble.HistGradientBoostingRegressor(
                loss="squared_error"
            ),
            param_grid=param_grid,
            scoring=sklearn.metrics.make_scorer(
                a6.training.metrics.turbine.calculate_nrmse,
                greater_is_better=False,
                power_rating=power_rating,
            ),
            # 10-fold CV
            cv=10,
            refit=True,
            n_jobs=n_jobs,
        )
        gs = gs.fit(X=X_train, y=y_train)

        start, end = min(turbine["time"].values), max(turbine["time"].values)
        dates = pd.date_range(start, end, freq="1d")

        results: list[Errors] = [
            _calculate_nmae_and_nrmse(
                start=start,
                end=end,
                gs=gs,
                weather_data=data,
                turbine=turbine,
                power_rating=power_rating,
                turbine_variables=turbine_variables,
                coordinates=coordinates,
            )
            for stard, end in itertools.pairwise(dates)
        ]

        nmae_da = xr.DataArray(
            [error.nmae for error in results],
            coords={"time": dates[:-1]},
            dims=["time"],
        )
        nrmse_da = xr.DataArray(
            [error.nrmse for error in results],
            coords={"time": dates[:-1]},
            dims=["time"],
        )
        errors = xr.Dataset(
            data_vars={"nmae": nmae_da, "nrmse": nrmse_da},
            coords=nmae_da.coords,
        )

        logger.info("Saving simulated forecast errors to %s", outfile)
        errors.to_netcdf(outfile)

        result[outfile] = errors

    return result


@dataclasses.dataclass
class Errors:
    nmae: float
    nrmse: float


def _calculate_nmae_and_nrmse(
    start: datetime.datetime,
    end: datetime.datetime,
    gs: sklearn.model_selection.GridSearchCV,
    weather_data: list[xr.DataArray],
    turbine: xr.Dataset,
    power_rating: float,
    turbine_variables: _variables.Turbine,
    coordinates: _coordinates.Coordinates,
) -> Errors:
    window = slice(start, end)
    logger.info("Evaluating model error for slice=%s", window)

    weather_forecast = [d.sel({coordinates.time: window}) for d in weather_data]
    X_forecast = a6.features.methods.reshape.sklearn.transpose(  # noqa: N806
        *weather_forecast
    )

    turbine_sub = turbine.sel({coordinates.time: window})[
        turbine_variables.production
    ]
    y_true = a6.features.methods.reshape.sklearn.transpose(turbine_sub)

    if y_true.size == 0:
        logger.warning(
            (
                "Emtpy production data for start=%s "
                "end=%s, setting errors to NaN"
            ),
            start,
            end,
        )
        return Errors(np.nan, np.nan)

    y_pred = gs.predict(X_forecast)

    nmae = a6.training.metrics.turbine.calculate_nmae(
        y_true=y_true, y_pred=y_pred, power_rating=power_rating
    )
    nrmse = a6.training.metrics.turbine.calculate_nrmse(
        y_true=y_true, y_pred=y_pred, power_rating=power_rating
    )
    return Errors(nmae=nmae, nrmse=nrmse)
