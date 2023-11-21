import argparse
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

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pressure-level-data",
    type=pathlib.Path,
)
parser.add_argument(
    "--model-level-data",
    type=pathlib.Path,
)
parser.add_argument(
    "--surface-level-data",
    type=pathlib.Path,
)
parser.add_argument(
    "--turbine-data-dir",
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

ForecastErrors = dict[pathlib.Path, xr.Dataset]


def simulate_forecast_errors(
    raw_args: list[str] | None = None,
) -> ForecastErrors:
    args = parser.parse_args(raw_args)

    turbine_files = sorted(list(args.turbine_data_dir.rglob("**/*.nc")))

    coordinates: _coordinates.Coordinates = _coordinates.Coordinates()
    model_variables: _variables.Model = _variables.Model()
    turbine_variables: _variables.Turbine = a6.datasets.variables.Turbine()

    ds_sfc = xr.open_dataset(args.surface_level_data)
    ds_ml = xr.open_dataset(args.model_level_data).sel({coordinates.level: 137})
    ds_pl = xr.open_dataset(args.pressure_level_data).sel(
        {coordinates.level: 1000}
    )

    result: ForecastErrors = {}

    for i, turbine_path in enumerate(turbine_files):
        logger.info(
            "Processing turbine %i/%i (path=%s)",
            i,
            len(turbine_files),
            turbine_path,
        )

        file_name = turbine_path.name.replace(".nc", "_forecast_errors.nc")
        outfile: pathlib.Path = args.results_dir / file_name

        if outfile.exists():
            logger.warning(
                "Skipping %s since outfile already exists at %s",
                turbine_path,
                outfile,
            )

        turbine = xr.open_dataset(turbine_path)

        power_rating = turbine_variables.read_power_rating(turbine)
        logger.info("Extracted power rating %i", power_rating)

        logger.info("Preprocessing turbine data")
        turbine = (
            a6.datasets.methods.turbine.clean_production_data(
                power_rating=power_rating,
                variables=turbine_variables,
            )
            >> a6.datasets.methods.turbine.resample_to_hourly_resolution(
                variables=turbine_variables,
                coordinates=coordinates,
            )
            >> a6.datasets.methods.select.select_latitude_longitude(
                latitude=0, longitude=0
            )
        ).apply_to(turbine)

        logger.info("Preprocessing surface level data")
        sfc = (
            a6.datasets.methods.turbine.get_closest_grid_point(
                turbine=turbine,
                coordinates=coordinates,
            )
            >> a6.datasets.methods.turbine.select_intersecting_time_steps(
                turbine=turbine, coordinates=coordinates
            )
            >> a6.datasets.methods.select.select_variables(
                variables=model_variables.sp
            )
        ).apply_to(ds_sfc)

        logger.info("Preprocessing model level data")
        ml = (
            a6.datasets.methods.turbine.get_closest_grid_point(
                turbine=turbine,
                coordinates=coordinates,
            )
            >> a6.datasets.methods.turbine.select_intersecting_time_steps(
                turbine=turbine, coordinates=coordinates
            )
            >> a6.datasets.methods.select.select_variables(
                variables=model_variables.t
            )
        ).apply_to(ds_ml)

        logger.info("Preprocessing pressure level data")
        pl = (
            a6.datasets.methods.turbine.get_closest_grid_point(
                turbine=turbine,
                coordinates=coordinates,
            )
            >> a6.datasets.methods.turbine.select_intersecting_time_steps(
                turbine=turbine, coordinates=coordinates
            )
            >> a6.features.methods.wind.calculate_wind_speed(
                variables=model_variables
            )
            >> a6.features.methods.wind.calculate_wind_direction_angle(
                variables=model_variables
            )
            >> a6.features.methods.time.calculate_fraction_of_day(
                coordinates=coordinates
            )
            >> a6.features.methods.time.calculate_fraction_of_year(
                coordinates=coordinates
            )
            >> a6.datasets.methods.select.select_variables(
                variables=[
                    model_variables.wind_speed,
                    model_variables.wind_direction,
                    model_variables.r,
                    "fraction_of_year",
                    "fraction_of_day",
                ]
            )
        ).apply_to(ds_pl)

        _, turbine = a6.datasets.methods.turbine.select_intersecting_time_steps(
            weather=ml,
            turbine=turbine,
            coordinates=coordinates,
            return_turbine=True,
            non_functional=True,
        )

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
        ) = sklearn.model_selection.train_test_split(X, y, train_size=2 / 3)

        logger.info(
            "Train dataset size is %i hours (~%i days)",
            y_train.size,
            y_train.size / 24,
        )

        logger.info("Fitting model with GridSearchCV")

        gs = sklearn.model_selection.GridSearchCV(
            estimator=ensemble.HistGradientBoostingRegressor(
                loss="squared_error"
            ),
            param_grid={
                "learning_rate": [0.03, 0.05, 0.07, 0.1],
                "l2_regularization": [0.0, 1.0, 3.0, 5.0, 7.0],
                "max_iter": [200, 300, 500],
                "max_depth": [15, 37, 63, 81],
                "min_samples_leaf": [23, 48, 101, 199],
            },
            scoring=sklearn.metrics.make_scorer(
                a6.training.metrics.turbine.calculate_nrmse,
                greater_is_better=False,
                power_rating=power_rating,
            ),
            # 10-fold CV
            cv=10,
            refit=True,
            n_jobs=-1 if args.parallel else int(a6.utils.get_cpu_count() / 2),
        )
        gs = gs.fit(X=X_train, y=y_train)

        start, end = min(turbine["time"].values), max(turbine["time"].values)
        dates = pd.date_range(start, end, freq="1d")

        nrmse_all = []
        nmae_all = []

        for start, end in itertools.pairwise(dates):
            window = slice(start, end)
            logger.info("Evaluating model error for slice=%s", window)

            weather_forecast = [d.sel({coordinates.time: window}) for d in data]
            X_forecast = (  # noqa: N806
                a6.features.methods.reshape.sklearn.transpose(*weather_forecast)
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
                nrmse_all.append(np.nan)
                nmae_all.append(np.nan)
                continue

            y_pred = gs.predict(X_forecast)

            nrmse = a6.training.metrics.turbine.calculate_nrmse(
                y_true=y_true, y_pred=y_pred, power_rating=power_rating
            )
            nmae = a6.training.metrics.turbine.calculate_nmae(
                y_true=y_true, y_pred=y_pred, power_rating=power_rating
            )
            nrmse_all.append(nrmse)
            nmae_all.append(nmae)

        nmae_da = xr.DataArray(
            nmae_all,
            coords={"time": dates[:-1]},
            dims=["time"],
        )
        nrmse_da = xr.DataArray(
            nrmse_all,
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
