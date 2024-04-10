import argparse
import dataclasses
import datetime
import logging
import pathlib
import os

import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.metrics
import sklearn.model_selection
import xarray as xr
import torch

import a6
import a6.datasets.coordinates as _coordinates
import a6.datasets.variables as _variables
import a6.utils as utils

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--turbine-data-dir",
    type=pathlib.Path,
    help="Directory of the turbine data are located, saved in netCDF format."
)
parser.add_argument(
    "--preprocessed-data-dir",
    type=pathlib.Path,
    help="Directory of the preprocessed turbine and weather data for training the models."
)
parser.add_argument(
    "--pca-kpca-path",
    type=pathlib.Path,
    default=None,
    help="Path to the file containing the PCA and kPCA LSWR labels."
)
parser.add_argument(
    "--pca-kpca-n-clusters",
    type=int,
    default=40,
    choices=[29, 40],
    help="Number of categories to use from the PCA-kPCA file."
)
parser.add_argument(
    "--gwl-path",
    type=pathlib.Path,
    default=None,
    help="Path to the file containing the DWD GWL labels."
)
parser.add_argument(
    "--dcv2-path",
    type=pathlib.Path,
    default=None,
    help="Path to the file containing the DCv2 LSWR labels."
)
parser.add_argument(
    "--random",
    type=bool,
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Train the model with random input labels."
        ""
        "Enabling this option will neglect given DWD, PCA, kPCA, and DCv2 labels."
    ),
)
parser.add_argument(
    "--train-size",
    type=int,
    default=365,
    help="Number of random days to use for the train set",
)
parser.add_argument(
    "--results-dir",
    type=pathlib.Path,
    help="Path to store the forecast error results."
)
parser.add_argument(
    "--parallel",
    type=bool,
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Parallelize workloads.",
)
parser.add_argument(
    "--testing",
    type=bool,
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Reduce the parameter space.",
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

    if not args.parallel:
        WORKER_ID = None
        logger.info(
            "'--no-parallel' passed, setting WORKER_ID=None to prevent "
            "parallel processing of turbine data"
        )
    else:
        WORKER_ID = int(os.getenv("SLURM_PROCID")) if "SLURM_PROCID" in os.environ else None
        logging.info(
            "'--parallel' passed, reading WORKER_ID=%s from SLURM_PROCID=%s ",
            WORKER_ID,
            os.getenv("SLURM_PROCID"),
        )

    utils.logging.create_logger(
        global_rank=WORKER_ID,
        local_rank=WORKER_ID,
        verbose=False,
    )

    turbine_files = a6.utils.paths.list_files(
        args.turbine_data_dir, pattern="**/*.nc", recursive=True
    )
    
    if WORKER_ID is not None and WORKER_ID >= len(turbine_files):
        logger.warning("Exiting: no file to process")
        return

    coordinates: _coordinates.Coordinates = _coordinates.Coordinates()
    turbine_variables: _variables.Turbine = a6.datasets.variables.Turbine()
    
    lswrs = [None]

    if args.gwl_path is not None:
        gwl = xr.open_dataset(args.gwl_path)
        lswrs.append(gwl["GWL"])
    
    if args.pca_kpca_path is not None:
        pca_kpca = xr.open_dataset(
            args.pca_kpca_path
        ).sel(
            k=args.pca_kpca_n_clusters
        )
        lswrs.extend([pca_kpca["PCA"], pca_kpca["kPCA"]])
        
    if args.dcv2_path is not None:
        dcv2 = xr.open_dataset(args.dcv2_path)
        lswrs.append(dcv2["DCv2"])

    if args.random:
        logger.info(
            "Using randomized LSWR labels for training (n_categories=%i)",
            args.pca_kpca_n_clusters,
        )
        dates = pd.date_range("2000-01-01", datetime.date.today(), freq="1D")
        random_labels = xr.DataArray(
            np.random.randint(args.pca_kpca_n_clusters, size=len(dates)),
            coords={coordinates.time: dates},
            dims=[coordinates.time],
            name="Random",
        )
        lswrs.append(random_labels)

    result: ForecastErrors = {}
    
    logger.info("Creating results directory if not exists at %s", args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

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
        outfile: pathlib.Path = (
            args.results_dir / f"{turbine_name}-forecast-errors.nc"
        )

        if outfile.exists():
            logger.warning(
                "Skipping %s since outfile already exists at %s",
                turbine_name,
                outfile,
            )
            continue

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

        start, end = min(turbine[coordinates.time].values), max(turbine[coordinates.time].values)

        # Convert time stamps to dates and create date range
        times_as_dates = a6.utils.times.time_steps_as_dates(turbine, coordinates=coordinates)
        start, end = min(times_as_dates), max(times_as_dates)
        dates = pd.date_range(start, end, freq="1d")

        # Train with minimum 50% of the number of days in the turbine data set,
        # but with a maximum of 365 days.
        train_size = min(0.5 * len(dates), args.train_size)

        train_time_steps, test_time_steps = a6.features.methods.selection.train_test_split_dates(
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
            lswrs,
            start,
            end,
            len(train_time_steps),
            len(test_time_steps),
        )

        forecast_errors = {}

        for lswr in lswrs:
            lswr_name = "Default" if lswr is None else lswr.name
            
            logger.info("Handling LSWR method %s", lswr_name)
            
            data = [ml[var] for var in ml.data_vars] + [sfc[var] for var in sfc.data_vars] + [pl[var] for var in pl.data_vars]
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

            data_train = [d.sel({coordinates.time: train_time_steps}) for d in data]
            data_test = [d.sel({coordinates.time: test_time_steps}) for d in data]

            production = turbine[turbine_variables.production]
            turbine_train = production.sel({coordinates.time: train_time_steps})
            turbine_test = production.sel({coordinates.time: test_time_steps})
                
            logger.info(
                "Preparing input data for variables %s", [d.name for d in data]
            )

            X = a6.features.methods.reshape.sklearn.transpose(*data_train)  # noqa: N806
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

            results: list[Errors] = a6.utils.parallelize.parallelize_with_futures(
                _calculate_nmae_and_nrmse,
                kwargs=[
                    dict(
                        date=date,
                        test_time_steps=test_time_steps,
                        gs=gs,
                        weather_data=data_test,
                        turbine=turbine_test,
                        power_rating=power_rating,
                        coordinates=coordinates,
                    )
                    for date in dates
                ]
            )
            
            forecast_errors[lswr_name] = results

        coords = {coordinates.time: dates, "lswr_method": list(forecast_errors.keys())}
        dims = [coordinates.time, "lswr_method"]

        nmae_da = xr.DataArray(
            _unpack_errors_per_method(forecast_errors, attr="nmae"),
            coords=coords,
            dims=dims,
        )
        nrmse_da = xr.DataArray(
            _unpack_errors_per_method(forecast_errors, attr="nrmse"),
            coords=coords,
            dims=dims,
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
    date: pd.Timestamp,
    test_time_steps: xr.DataArray,
    gs: sklearn.model_selection.GridSearchCV,
    weather_data: list[xr.DataArray],
    turbine: xr.Dataset,
    power_rating: float,
    coordinates: _coordinates.Coordinates,
) -> Errors:
    logger.debug("Evaluating model error for %s", date)

    if date not in test_time_steps:
        logger.info("Date %s not in test set, setting errors to NaN", date)
        return Errors(np.nan, np.nan)

    # Need to select turbine before weather data,
    # because methods applied on weather data may fail 
    # if no turbine data are available for the given date.
    turbine_sub = a6.datasets.methods.select.select_for_date(
        turbine, date=date
    )
    y_true = a6.features.methods.reshape.sklearn.transpose(turbine_sub)

    if y_true.size < 3:
        logger.warning(
            (
                "Less than 3 time steps for production data for date=%s, "
                "setting errors to NaN"
            ),
            date,
        )
        return Errors(np.nan, np.nan)

    weather_forecast = [a6.datasets.methods.select.select_for_date(d, date=date) for d in weather_data]
    X_forecast = a6.features.methods.reshape.sklearn.transpose(  # noqa: N806
        *weather_forecast
    )

    y_pred = gs.predict(X_forecast)

    nmae = a6.training.metrics.turbine.calculate_nmae(
        y_true=y_true, y_pred=y_pred, power_rating=power_rating
    )
    nrmse = a6.training.metrics.turbine.calculate_nrmse(
        y_true=y_true, y_pred=y_pred, power_rating=power_rating
    )
    return Errors(nmae=nmae, nrmse=nrmse)

def _unpack_errors_per_method(
    errors: list[ForecastErrors], attr: str
) -> list[list[float]]:
    return [
        [getattr(error, attr) for error in method]
        for method in zip(*errors.values())
    ]