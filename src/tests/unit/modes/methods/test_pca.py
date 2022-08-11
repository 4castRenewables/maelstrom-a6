import numpy as np
import pytest
import sklearn.decomposition as decomposition
import xarray as xr


@pytest.fixture(params=[decomposition.PCA, decomposition.IncrementalPCA])
def method(request):
    return request.param


def test_spatio_temporal_pca(ds, pcas):
    # da has n = 5 time steps on a (10, 10) grid, hence PCs must be of shape
    # (5, 100)
    assert pcas.components.shape == (5, 100)
    assert pcas.components_in_original_shape.shape == (5, 10, 10)
    assert pcas.components_varimax_rotated.shape == (5, 100)
    assert pcas.explained_variance.shape == (5,)
    assert pcas.explained_variance_ratio.shape == (5,)
    assert pcas.cumulative_variance_ratio.shape == (5,)
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


class TestPCA:
    def test_transform(self, pca):
        data: xr.DataArray = pca._original_reshaped
        sklearn_result: np.ndarray = pca._pca.transform(data.data)

        result: xr.DataArray = pca.transform()

        np.testing.assert_equal(result.data, sklearn_result)

    def test_inverse_transform(self, da, pca):
        sklearn_transformed = pca._pca.transform(pca._original_reshaped)
        sklearn_result = pca._pca.inverse_transform(sklearn_transformed)

        transformed = pca.transform()
        result = pca.inverse_transform(transformed, in_original_shape=False)

        np.testing.assert_allclose(np.abs(result.data), np.abs(sklearn_result))
