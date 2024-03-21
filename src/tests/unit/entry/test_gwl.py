import a6.entry as entry


def test_main(era5_path, gwl_path):
    # Don't select DWD area for testing due to failure.
    entry.gwl.main(
        epochs=1,
        data_path=era5_path,
        gwl_path=gwl_path,
        select_dwd_area=False,
        architecture="ignored-for-testing",
        testing=True,
        log_to_mlflow=False,
    )
