import a6.plotting.modes as modes


def test_plot_geopotential_height_contours(pl_ds):
    modes.plot_geopotential_height_contours(
        pl_ds["z"].sel(level=500).isel(time=0)
    )
