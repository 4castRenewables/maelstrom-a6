import pytest
import sklearn.decomposition as decomposition
import xarray as xr


@pytest.fixture(params=[decomposition.PCA, decomposition.IncrementalPCA])
def method(request):
    return request.param


def test_spatio_temporal_principal_component_analysis(ds, pcas):
    # da has n = 5 time steps on a (10, 10) grid, hence PCs must be of shape
    # (5, 100)
    assert pcas.components.shape == (5, 100)
    assert pcas.components_in_original_shape.shape == (5, 10, 10)
    assert pcas.components_varimax_rotated.shape == (5, 100)
    assert pcas.eigenvalues.shape == (5,)
    assert pcas.loadings.shape == (5, 100)
    assert pcas.variance_ratios.shape == (5,)
    assert pcas.cumulative_variance_ratios.shape == (5,)
    assert pcas.transform().shape == (5, 5)
    assert pcas.transform(n_components=2).shape == (5, 2)
    assert pcas.transform_with_varimax_rotation().shape == (5, 5)
    assert pcas.transform_with_varimax_rotation(n_components=2).shape == (5, 2)
    assert (
        pcas.number_of_components_sufficient_for_variance_ratio(
            variance_ratio=0.5
        )
        == 1
    )
    assert pcas.components_sufficient_for_variance_ratio(0.5).shape == (1, 100)
    assert pcas.inverse_transform(
        xr.DataArray([1, 2, 3]), n_components=3
    ).shape == (10, 10)
