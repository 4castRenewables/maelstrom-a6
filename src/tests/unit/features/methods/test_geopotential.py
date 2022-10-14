import a6.features.methods.geopotential as geopotential


def test_calculate_geopotential_height(pl_ds):
    geopotential.calculate_geopotential_height(pl_ds["z"])
