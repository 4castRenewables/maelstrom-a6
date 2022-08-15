import numpy as np
import xarray as xr


def test_spatio_temporal_pca(ds, single_variable_pca):
    # da has n = 5 time steps on a (10, 10) grid, hence PCs must be of shape
    # (5, 100)
    assert single_variable_pca.components.shape == (5, 100)
    assert single_variable_pca.components_in_original_shape.shape == (5, 10, 10)
    assert single_variable_pca.components_varimax_rotated.shape == (5, 100)
    assert single_variable_pca.explained_variance.shape == (5,)
    assert single_variable_pca.explained_variance_ratio.shape == (5,)
    assert single_variable_pca.cumulative_variance_ratio.shape == (5,)
    assert single_variable_pca.transform().shape == (5, 5)
    assert single_variable_pca.transform(n_components=2).shape == (5, 2)
    assert single_variable_pca.transform_with_varimax_rotation().shape == (5, 5)
    assert single_variable_pca.transform_with_varimax_rotation(
        n_components=2
    ).shape == (5, 2)
    assert (
        single_variable_pca.number_of_components_sufficient_for_variance_ratio(
            variance_ratio=0.5
        )
        == 1
    )
    assert single_variable_pca.components_sufficient_for_variance_ratio(
        0.5
    ).shape == (1, 100)
    assert single_variable_pca.inverse_transform(
        xr.DataArray([1, 2, 3]), n_components=3
    ).shape == (10, 10)


def test_multi_variable_spatio_temporal_pca(multi_variable_pca):

    # da has n = 5 time steps for 2 variables on a (10, 10) grid, hence PCs
    # must be of shape (5, 2*100)
    assert multi_variable_pca.components.shape == (5, 200)
    assert multi_variable_pca.components_in_original_shape.shape == (
        5,
        10,
        10,
        2,
    )
    assert multi_variable_pca.components_varimax_rotated.shape == (5, 200)
    assert multi_variable_pca.explained_variance.shape == (5,)
    assert multi_variable_pca.explained_variance_ratio.shape == (5,)
    assert multi_variable_pca.cumulative_variance_ratio.shape == (5,)
    assert multi_variable_pca.transform().shape == (5, 5)
    assert multi_variable_pca.transform(n_components=2).shape == (5, 2)
    assert multi_variable_pca.transform_with_varimax_rotation().shape == (5, 5)
    assert multi_variable_pca.transform_with_varimax_rotation(
        n_components=2
    ).shape == (5, 2)
    assert (
        multi_variable_pca.number_of_components_sufficient_for_variance_ratio(
            variance_ratio=0.5
        )
        == 1
    )
    assert multi_variable_pca.components_sufficient_for_variance_ratio(
        0.5
    ).shape == (1, 200)
    assert multi_variable_pca.inverse_transform(
        xr.DataArray([1, 2, 3]), n_components=3
    ).shape == (10, 10, 2)


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
