import lifetimes.modes.methods.pca as _pca
import pytest
import xarray as xr
from sklearn import decomposition


@pytest.fixture(params=[decomposition.PCA, decomposition.IncrementalPCA])
def method(request):
    return request.param


def test_spatio_temporal_pca(ds, pcas):
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


def test_multi_variable_spatio_temporal_pca(da):
    # Create multi-variable dataset
    ds = xr.Dataset(
        data_vars={"ellipse_1": da, "ellipse_2": da},
        coords=da.coords,
        attrs=da.attrs,
    )
    pca = _pca.spatio_temporal_pca(
        data=ds,
        time_coordinate="time",
        latitude_coordinate="lat",
        x_coordinate="lat",
        y_coordinate="lon",
        variance_ratio=None,
        pca_method=decomposition.PCA,
    )

    # da has n = 5 time steps for 2 variables on a (10, 10) grid, hence PCs
    # must be of shape (5, 2*100)
    assert pca.components.shape == (5, 200)
    assert pca.components_in_original_shape.shape == (5, 10, 10, 2)
    assert pca.components_varimax_rotated.shape == (5, 200)
    assert pca.eigenvalues.shape == (5,)
    assert pca.loadings.shape == (5, 200)
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
    assert pca.components_sufficient_for_variance_ratio(0.5).shape == (1, 200)
    assert pca.inverse_transform(
        xr.DataArray([1, 2, 3]), n_components=3
    ).shape == (10, 10, 2)
