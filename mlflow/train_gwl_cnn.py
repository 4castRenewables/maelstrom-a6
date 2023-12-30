import pathlib

import a6

if __name__ == "__main__":
    path = pathlib.Path("/home/fabian/data/era5/nc/era5_pl_1988_1999_12.nc")
    gwl_path = pathlib.Path(
        "/home/fabian/work/maelstrom/a6/src/tests/data/gwl.nc"
    )
    a6.entry.gwl.main(
        epochs=1,
        data_path=path,
        gwl_path=gwl_path,
        select_dwd_area=True,
        testing=False,
        log_to_mlflow=False,
    )
