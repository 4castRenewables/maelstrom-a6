import argparse
import dataclasses
import joblib
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

a6.utils.logging.create_logger(
    global_rank=0,
    local_rank=0,
    verbose=False,
)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--turbine-id",
    type=int,
    default=4,
    help="ID of the turbine to simulate the forecast for.",
)
parser.add_argument(
    "--testing",
    type=bool,
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Reduce the parameter space.",
)

args = parser.parse_args()

WORKER_ID = args.turbine_id

coordinates: _coordinates.Coordinates = _coordinates.Coordinates()
variables = a6.datasets.variables.Model()
turbine_variables: _variables.Turbine = a6.datasets.variables.Turbine()

turbine_data_dir = pathlib.Path(
    "/p/home/jusers/emmerich1/juwels/data/production"
)
preprocessed_data_dir = pathlib.Path(
    "/p/home/jusers/emmerich1/juwels/data/production-preprocessed"
)
results_dir = pathlib.Path(
    "/p/project/hclimrep/emmerich1/data/forecast-errors"
)
turbine_files = a6.utils.paths.list_files(
    turbine_data_dir, pattern="**/*.nc", recursive=True
)

results = xr.open_dataset(
    "/p/project/hclimrep/emmerich1/data/pca_kmeans_lswrs_30_40.nc"
)
n_lswr_categories = 30
results_pca = results.sel(k=n_lswr_categories)
gwl = xr.open_dataset("/p/home/jusers/emmerich1/juwels/code/a6/src/tests/data/gwl.nc")
dcv2 = xr.open_dataset("/p/project/hclimrep/emmerich1/data/dcv2-lswrs.nc")

forecast_inputs = [
    None,
    dcv2["DCv2"],
    results_pca["PCA"],
    gwl["GWL"],
]

@dataclasses.dataclass
class Errors:
    nmae: float
    nrmse: float


def _create_forecast(
    date: pd.Timestamp,
    gs: sklearn.model_selection.GridSearchCV,
    weather_data: list[xr.DataArray],
    turbine: xr.Dataset,
    turbine_variables: _variables.Turbine,
) -> tuple[np.ndarray, pd.Timestamp]:
    logger.debug("Creating forecast for %s", date)

    turbine_sub = a6.datasets.methods.select.select_for_date(
        turbine, date=date
    )[turbine_variables.production]
    y_true = a6.features.methods.reshape.sklearn.transpose(turbine_sub)

    if y_true.size < 24:
        logger.warning(
            (
                "Less than 24 time steps for production data for date=%s, "
                "returning empty array"
            ),
            date,
        )
        return np.array([]), date

    weather_forecast = [
        a6.datasets.methods.select.select_for_date(d, date=date)
        for d in weather_data
    ]
    X_forecast = a6.features.methods.reshape.sklearn.transpose(  # noqa: N806
        *weather_forecast
    )

    y_pred = gs.predict(X_forecast)
    return y_pred, date

if WORKER_ID is not None and WORKER_ID >= len(turbine_files):
    logger.warning("Exiting: no file to process")
    raise RuntimeError()


result = {}

for i, turbine_path in enumerate(turbine_files):
    if WORKER_ID is not None and i != WORKER_ID:
        continue

    logger.info(
        "Processing turbine %i/%i (path=%s)",
        i,
        len(turbine_files),
        turbine_path,
    )

    turbine_name = turbine_path.name.replace(".nc", "")

    turbine_path: pathlib.Path = (
        preprocessed_data_dir / f"{turbine_name}/turbine.nc"
    )
    pl_path: pathlib.Path = preprocessed_data_dir / f"{turbine_name}/pl.nc"
    ml_path: pathlib.Path = preprocessed_data_dir / f"{turbine_name}/ml.nc"
    sfc_path: pathlib.Path = preprocessed_data_dir / f"{turbine_name}/sfc.nc"

    logger.info("Reading preprocessed data")

    try:
        turbine = xr.open_dataset(turbine_path)
    except FileNotFoundError:
        logger.exception(
            "No preprocessed data for turbine %s found in %s",
            turbine_name,
            turbine_path,
        )
        continue

    pl = xr.open_dataset(pl_path)
    ml = xr.open_dataset(ml_path)
    sfc = xr.open_dataset(sfc_path)

    power_rating = turbine_variables.read_power_rating(turbine)
    logger.info("Extracted power rating %i", power_rating)

    # Convert time stamps to dates and create date range
    times_as_dates = a6.utils.times.time_steps_as_dates(
        turbine, coordinates=coordinates
    )
    start, end = min(times_as_dates), max(times_as_dates)
    dates = pd.date_range(start, end, freq="1d")

    # Train with minimum 70% of the number of days in the turbine data set,
    # but with a maximum of 365 days.
    train_size = min(0.7 * len(dates), 365)

    (
        train_time_steps,
        test_time_steps,
    ) = a6.features.methods.selection.train_test_split_dates(
        turbine[coordinates.time],
        # Turbine data has frequency of hours, hence multiply by 24
        # to achieve train set size equivalent to 365 days.
        train_size=int(train_size * 24),
    )

    logger.info(
        (
            "Simulating forecast errors for LSWRS %s for date range "
            "%s to %s with %i/%i train/test samples (hours)"
        ),
        forecast_inputs,
        start,
        end,
        len(train_time_steps),
        len(test_time_steps),
    )

    forecasts = {}

    for lswr in forecast_inputs:
        lswr_name = "Default" if lswr is None else lswr.name

        logger.info("Handling LSWR %s", lswr_name)

        outfile: pathlib.Path = (
            results_dir / f"{turbine_name}-forecast-errors-lswr-{lswr_name}.nc"
        )

        if outfile.exists():
            logger.warning(
                "Skipping %s since outfile already exists at %s",
                turbine_path,
                outfile,
            )

        data = (
            [ml[var] for var in ml.data_vars]
            + [sfc[var] for var in sfc.data_vars]
            + [pl[var] for var in pl.data_vars]
        )
        categorical_features = [False for _ in enumerate(data)]

        if lswr is not None:
            turbine_time_steps = turbine[coordinates.time]
            lswr_labels = lswr.sel(time=turbine_time_steps, method="pad")
            # Must override time coordinates of result, because due to "pad"
            # duplicate indexes are returned (the same index for every
            # hour of the day).
            lswr_labels[coordinates.time] = turbine_time_steps
            data.append(lswr_labels)
            categorical_features.append(True)

        logger.info(
            "Preparing input data for variables %s", [d.name for d in data]
        )

        data_train = [
            d.sel({coordinates.time: train_time_steps}) for d in data
        ]
        data_test = [
            d.sel({coordinates.time: test_time_steps}) for d in data
        ]

        production = turbine[turbine_variables.production]
        turbine_train = production.sel({coordinates.time: train_time_steps})
        turbine_test = production.sel({coordinates.time: test_time_steps})

        logger.info(
            "Preparing input data for variables %s", [d.name for d in data]
        )

        X = a6.features.methods.reshape.sklearn.transpose(  # noqa: N806
            *data_train
        )  # noqa: N806
        y = a6.features.methods.reshape.sklearn.transpose(turbine_train)

        logger.info(
            "Train dataset size is %i hours (~%i days)",
            y.size,
            y.size // 24,
        )


        if args.testing:
            param_grid = {"learning_rate": [0.1]}
            n_jobs = a6.utils.get_cpu_count() // 2
        else:
            param_grid = {
                "learning_rate": [0.03, 0.05, 0.07, 0.1],
                "l2_regularization": [0.0, 1.0, 3.0, 5.0, 7.0],
                "max_iter": [200, 300, 500],
                "max_depth": [15, 37, 63, 81],
                "min_samples_leaf": [23, 48, 101, 199],
                "categorical_features": [categorical_features],
            }
            n_jobs = a6.utils.get_cpu_count()

        logger.info(
            "Fitting model with GridSearchCV n_jobs=%s, param_grid=%s",
            n_jobs,
            param_grid,
        )

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
        gs = gs.fit(X=X, y=y.ravel())

        for date in dates:
            if lswr_name in forecasts:
                break

            forecast, date = _create_forecast(
                date=date,
                gs=gs,
                weather_data=data,
                turbine=turbine,
                turbine_variables=turbine_variables,
            )

            if forecast.size != 0:
                forecasts[lswr_name] = (forecast, date)

joblib.dump(
    forecasts, "/p/project/hclimrep/emmerich1/data/forecasts-per-method.joblib"
)