import pathlib

import xarray as xr

import a6.entry.preprocess as preprocess

FILE_PATH = pathlib.Path(__file__).parent
DATA_DIR = FILE_PATH / "../../data"
ML_FILE = FILE_PATH / "../../data/ml_20200101-02-turbine-matched.nc"
PL_FILE = FILE_PATH / "../../data/pl_20200101-02-turbine-matched.nc"
SFC_FILE = FILE_PATH / "../../data/sfc_20200101-02-turbine-matched.nc"
TURBINE_DIR = DATA_DIR / "turbine"


def test_create_turbine_model_features(tmp_path):
    args = [
        "--pressure-level-data",
        PL_FILE.as_posix(),
        "--model-level-data",
        ML_FILE.as_posix(),
        "--surface-level-data",
        SFC_FILE.as_posix(),
        "--turbine-data-dir",
        TURBINE_DIR.as_posix(),
        "--output-dir",
        tmp_path.as_posix(),
    ]

    result = preprocess.create_turbine_model_features(raw_args=args)

    # Ensure files were saved to disk.
    for path, res in result.items():
        xr.open_dataset(path)
