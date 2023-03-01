import pathlib

import a6.datasets.methods.convert as convert


def test_convert_fields_to_grayscale_images(
    pl_ds,
    tmp_path,
):
    expected = [
        pathlib.Path(tmp_path / "pl_2020-12-01T00:00_level_500_t_r_u.tif"),
        pathlib.Path(tmp_path / "pl_2020-12-01T01:00_level_500_t_r_u.tif"),
    ]

    convert.convert_fields_to_grayscale_images(
        pl_ds.sel(level=500).isel(time=slice(2)),
        variables=["t", "r", "u"],
        path=tmp_path,
        filename_prefix="pl_",
    )

    assert all(exp.exists() for exp in expected)
