import functools

import lifetimes.modes


def test_determine_modes(ds):
    modes = [lifetimes.modes.Modes(feature=ds["ellipse"])]
    pca_partial_method = functools.partial(
        lifetimes.modes.methods.spatio_temporal_principal_component_analysis,
        time_coordinate="time",
        latitude_coordinate="lat",
        x_coordinate="lat",
        y_coordinate="lon",
        variance_ratio=None,
    )
    [pca] = lifetimes.modes.determine_modes(modes=modes, method=pca_partial_method)

    assert isinstance(pca, lifetimes.modes.methods.PCA)
