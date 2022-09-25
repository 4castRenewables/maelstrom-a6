import functools

import a6.modes


def test_determine_modes(ds, coordinates):
    modes = [a6.modes.Modes(feature=ds["ellipse"])]
    pca_partial_method = functools.partial(
        a6.modes.methods.spatio_temporal_pca,
        coordinates=coordinates,
        x_coordinate="lat",
        y_coordinate="lon",
    )
    [pca] = a6.modes.determine_modes(modes=modes, method=pca_partial_method)

    assert isinstance(pca, a6.modes.methods.PCA)
