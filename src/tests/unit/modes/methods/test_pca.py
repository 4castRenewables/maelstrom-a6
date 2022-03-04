import lifetimes.modes.methods.pca as _pca
import pytest
from sklearn import decomposition


@pytest.fixture(params=[decomposition.PCA, decomposition.IncrementalPCA])
def method(request):
    return request.param


def test_spatio_temporal_principal_component_analysis(ds, method):
    da = ds["ellipse"]
    pca = _pca.spatio_temporal_principal_component_analysis(
        data=da,
        time_coordinate="time",
        latitude_coordinate="lat",
        x_coordinate="lat",
        y_coordinate="lon",
        variance_ratio=None,
        pca_method=method,
    )

    # da has n = 5 time steps on a (10, 10) grid, hence PCs must be of shape
    # (5, 100)
    assert pca.components.shape == (5, 100)
    assert pca.components_in_original_shape.shape == (5, 10, 10)
    assert pca.components_varimax_rotated.shape == (5, 100)
    assert pca.eigenvalues.shape == (5,)
    assert pca.loadings.shape == (5, 100)
    assert pca.variance_ratios.shape == (5,)
    assert pca.cumulative_variance_ratios.shape == (5,)
    assert pca.transform().shape == (5, 5)
    assert pca.transform(n_components=2).shape == (5, 2)
    assert pca.transform_with_varimax_rotation().shape == (5, 5)
    assert pca.transform_with_varimax_rotation(n_components=2).shape == (5, 2)
    assert (
        pca.number_of_components_sufficient_for_variance_ratio(
            variance_ratio=0.5
        )
        == 1
    )
