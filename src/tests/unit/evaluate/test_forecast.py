import pathlib

import xarray as xr

import a6.evaluation.forecast as forecast

FILE_PATH = pathlib.Path(__file__).parent
DATA_DIR = FILE_PATH / "../../data"
ML_FILE = FILE_PATH / "../../data/ml_20200101-02-turbine-matched.nc"
PL_FILE = FILE_PATH / "../../data/pl_20200101-02-turbine-matched.nc"
SFC_FILE = FILE_PATH / "../../data/sfc_20200101-02-turbine-matched.nc"
TURBINE_DIR = DATA_DIR / "turbine"


def test_simulate_forecast_errors(tmp_path):
    args = [
        "--pressure-level-data",
        PL_FILE.as_posix(),
        "--model-level-data",
        ML_FILE.as_posix(),
        "--surface-level-data",
        SFC_FILE.as_posix(),
        "--turbine-data-dir",
        TURBINE_DIR.as_posix(),
        "--results-dir",
        tmp_path.as_posix(),
        "--no-parallel",
    ]

    result = forecast.simulate_forecast_errors(raw_args=args)

    for path, res in result.items():
        exp = xr.open_dataset(path)

        xr.testing.assert_equal(res, exp)
