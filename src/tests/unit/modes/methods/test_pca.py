import numpy as np
import pytest
import xarray as xr


class TestPCA:
    @pytest.mark.parametrize(
        ("pca_", "expected"),
        [
            # n = 5 time steps on a (10, 10) grid, hence PCs must be of shape
            # (5, 100)
            ("single_variable_pca", (5, 100)),
            # n = 5 time steps for 2 variables on a (10, 10) grid, hence PCs
            # must be of shape (5, 2*100)
            ("multi_variable_pca", (5, 200)),
        ],
    )
    def test_components(self, request, pca_, expected):
        pca_ = request.getfixturevalue(pca_)

        result = pca_.components

        assert result.shape == expected

    @pytest.mark.parametrize(
        ("pca_", "expected"),
        [
            ("single_variable_pca", (5, 100)),
            ("multi_variable_pca", (5, 200)),
        ],
    )
    def test_components_varimax_rotated(self, request, pca_, expected):
        pca_ = request.getfixturevalue(pca_)

        result = pca_.components_varimax_rotated

        assert result.shape == expected

    @pytest.mark.parametrize(
        ("pca_", "expected_n_vars", "expected_shape"),
        [
            ("single_variable_pca", 1, (5, 10, 10)),
            ("multi_variable_pca", 2, (5, 10, 10)),
        ],
    )
    def test_components_in_original_shape(
        self, request, pca_, expected_n_vars, expected_shape
    ):
        pca_ = request.getfixturevalue(pca_)

        result = pca_.components_in_original_shape

        result_shape = tuple(result.sizes.values())

        result_n_vars = len(result.data_vars)
        assert result_n_vars == expected_n_vars
        assert result_shape == expected_shape

    @pytest.mark.parametrize(
        ("pca_", "expected"),
        [
            ("single_variable_pca", (5,)),
            ("multi_variable_pca", (5,)),
        ],
    )
    def test_explained_variance(self, request, pca_, expected):
        pca_ = request.getfixturevalue(pca_)

        result = pca_.explained_variance

        assert result.shape == expected

    @pytest.mark.parametrize(
        ("pca_", "expected"),
        [
            ("single_variable_pca", (5,)),
            ("multi_variable_pca", (5,)),
        ],
    )
    def test_explained_variance_ratio(self, request, pca_, expected):
        pca_ = request.getfixturevalue(pca_)

        result = pca_.explained_variance_ratio

        assert result.shape == expected

    @pytest.mark.parametrize(
        ("pca_", "expected"),
        [
            ("single_variable_pca", (5,)),
            ("multi_variable_pca", (5,)),
        ],
    )
    def test_cumulative_variance_ratio(self, request, pca_, expected):
        pca_ = request.getfixturevalue(pca_)

        result = pca_.cumulative_variance_ratio

        assert result.shape == expected

    @pytest.mark.parametrize(
        ("pca_", "variance_ratio", "expected"),
        [
            ("single_variable_pca", 0.5, 1),
            ("multi_variable_pca", 0.5, 1),
        ],
    )
    def test_number_of_components_sufficient_for_variance_ratio(
        self, request, pca_, variance_ratio, expected
    ):
        pca_ = request.getfixturevalue(pca_)

        result = pca_.number_of_components_sufficient_for_variance_ratio(
            variance_ratio
        )

        assert result == expected

    @pytest.mark.parametrize(
        ("pca_", "variance_ratio", "expected"),
        [
            ("single_variable_pca", 0.5, (1, 100)),
            ("multi_variable_pca", 0.5, (1, 200)),
        ],
    )
    def test_components_sufficient_for_variance_ratio(
        self, request, pca_, variance_ratio, expected
    ):
        pca_ = request.getfixturevalue(pca_)

        result = pca_.components_sufficient_for_variance_ratio(variance_ratio)

        assert result.shape == expected

    @pytest.mark.parametrize(
        ("pca_", "n_components", "expected"),
        [
            ("single_variable_pca", None, (5, 5)),
            ("single_variable_pca", 2, (5, 2)),
            ("multi_variable_pca", None, (5, 5)),
            ("multi_variable_pca", 2, (5, 2)),
        ],
    )
    def test_transform_shape(self, request, pca_, n_components, expected):
        pca_ = request.getfixturevalue(pca_)

        result = pca_.transform(n_components)

        assert result.shape == expected

    def test_transform(self, pca):
        data: xr.DataArray = pca._original_reshaped
        sklearn_result: np.ndarray = pca._pca.transform(data.data)

        result: xr.DataArray = pca.transform()

        np.testing.assert_equal(result.data, sklearn_result)

    @pytest.mark.parametrize(
        ("pca_", "n_components", "expected"),
        [
            ("single_variable_pca", None, (5, 5)),
            ("single_variable_pca", 2, (5, 2)),
            ("multi_variable_pca", None, (5, 5)),
            ("multi_variable_pca", 2, (5, 2)),
        ],
    )
    def test_transform_with_varimax_rotation(
        self, request, pca_, n_components, expected
    ):
        pca_ = request.getfixturevalue(pca_)

        result = pca_.transform_with_varimax_rotation(n_components)

        assert result.shape == expected

    @pytest.mark.parametrize(
        ("pca_", "data", "n_components", "expected_n_vars", "expected_shape"),
        [
            ("single_variable_pca", xr.DataArray([1, 2, 3]), 3, 1, (10, 10)),
            ("multi_variable_pca", xr.DataArray([1, 2, 3]), 3, 2, (10, 10)),
        ],
    )
    def test_inverse_transform_shape(
        self, request, pca_, data, n_components, expected_n_vars, expected_shape
    ):
        pca_ = request.getfixturevalue(pca_)

        result = pca_.inverse_transform(data, n_components=n_components)
        result_shape = tuple(result.sizes.values())

        result_n_vars = len(result.data_vars)
        assert result_n_vars == expected_n_vars
        assert result_shape == expected_shape

    def test_inverse_transform(self, da, pca):
        sklearn_transformed = pca._pca.transform(pca._original_reshaped)
        sklearn_result = pca._pca.inverse_transform(sklearn_transformed)

        transformed = pca.transform()
        result = pca.inverse_transform(transformed, in_original_shape=False)
        [var] = tuple(result.data_vars)
        result_da = result[var]

        np.testing.assert_allclose(
            np.abs(result_da.data), np.abs(sklearn_result)
        )
