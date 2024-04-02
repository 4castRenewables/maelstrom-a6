import argparse
import dataclasses
import datetime
import itertools
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
)
parser.add_argument(
    "--preprocessed-data-dir",
    type=pathlib.Path,
)
parser.add_argument(
    "--pca-kpca-path",
    type=pathlib.Path,
    default=None,
)
parser.add_argument(
    "--pca-kpca-n-clusters",
    type=int,
    default=40,
)
parser.add_argument(
    "--gwl-path",
    type=pathlib.Path,
    default=None,
)
parser.add_argument(
    "--dcv2-path",
    type=pathlib.Path,
    default=None,
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
    
    if args.pca_kpca_path is not None:
        pca_kpca = xr.open_dataset(
            args.pca_kpca_path
        ).sel(
            k=args.pca_kpca_n_clusters
        )
        lswrs.extend([pca_kpca["PCA"], pca_kpca["kPCA"]])
        
    if args.gwl_path is not None:
        gwl = xr.open_dataset(args.gwl_path)
        lswrs.append(gwl["GWL"])
        
    if args.dcv2_path is not None:
        dcv2 = xr.open_dataset(args.dcv2_path)
        lswrs.append(dcv2["DCv2"])

    result: ForecastErrors = {}
    
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

        start, end = min(turbine["time"].values), max(turbine["time"].values)

        # Convert time stamps to dates and create date range
        times_as_dates = a6.utils.times.time_steps_as_dates(turbine, coordinates=coordinates)
        start, end = min(times_as_dates), max(times_as_dates)
        dates = pd.date_range(start, end, freq="1d")

        logger.info(
            "Simulating forecast errors for LSWRS %s for date range %s to %s",
            lswrs,
            start,
            end,
        )

        forecast_errors = {}

        for lswr in lswrs:
            lswr_name = "none" if lswr is None else lswr.name
            
            logger.info("Handling LSWR %s", lswr_name)
            
            data = [ml[var] for var in ml.data_vars] + [sfc[var] for var in sfc.data_vars] + [pl[var] for var in pl.data_vars]
            categorical_features = [False for _ in enumerate(data)]
            
            if lswr is not None:
                lswr_labels = lswr.sel(time=turbine[coordinates.time], method="pad")
                data.append(lswr_labels)
                categorical_features.append(True)
                
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
            ) = sklearn.model_selection.train_test_split(X, y, train_size=1/3)

            logger.info(
                "Train dataset size is %i hours (~%i days)",
                y_train.size,
                y_train.size // 24,
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
            gs = gs.fit(X=X_train, y=y_train.ravel())

            results: list[Errors] = a6.utils.parallelize.parallelize_with_futures(
                _calculate_nmae_and_nrmse,
                kwargs=[
                    dict(
                        date=date,
                        gs=gs,
                        weather_data=data,
                        turbine=turbine,
                        power_rating=power_rating,
                        turbine_variables=turbine_variables,
                        coordinates=coordinates,
                    )
                    for date in dates
                ]
            )
            
            forecast_errors[lswr_name] = results

        coords = {"time": dates, "lswr_method": list(forecast_errors.keys())}
        dims = ["time", "lswr_method"]

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
    date: datetime.datetime,
    gs: sklearn.model_selection.GridSearchCV,
    weather_data: list[xr.DataArray],
    turbine: xr.Dataset,
    power_rating: float,
    turbine_variables: _variables.Turbine,
    coordinates: _coordinates.Coordinates,
) -> Errors:
    logger.debug("Evaluating model error for %s", date)

    weather_forecast = [a6.datasets.methods.select.select_for_date(d, date=date) for d in weather_data]
    X_forecast = a6.features.methods.reshape.sklearn.transpose(  # noqa: N806
        *weather_forecast
    )

    turbine_sub = a6.datasets.methods.select.select_for_date(
        turbine, date=date
    )[turbine_variables.production]
    y_true = a6.features.methods.reshape.sklearn.transpose(turbine_sub)

    if y_true.size < 6:
        logger.warning(
            (
                "Less than 6 time steps for production data for date=%s, "
                "setting errors to NaN"
            ),
            date,
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

def _unpack_errors_per_method(
    errors: list[ForecastErrors], attr: str
) -> list[list[float]]:
    return [
        [getattr(error, attr) for error in method]
        for method in zip(*errors.values())
    ]