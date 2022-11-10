import hdbscan
import pytest

import a6.studies.hyperparameters as _hyperparameters


class TestHyperParamers:
    @pytest.mark.parametrize(
        ("hyperparameters", "expected"),
        [
            (
                _hyperparameters.HyperParameters(
                    n_components_start=1,
                    cluster_arg="test-arg",
                    cluster_start=2,
                    cluster_end=3,
                ),
                [(1, 2), (1, 3)],
            ),
            (
                _hyperparameters.HyperParameters(
                    n_components_start=1,
                    n_components_end=2,
                    cluster_arg="test-arg",
                    cluster_start=2,
                ),
                [(1, 2), (2, 2)],
            ),
            (
                _hyperparameters.HyperParameters(
                    n_components_start=1,
                    n_components_end=2,
                    cluster_arg="test-arg",
                    cluster_start=2,
                    cluster_end=3,
                ),
                [(1, 2), (1, 3), (2, 2), (2, 3)],
            ),
        ],
    )
    def test_to_range(self, hyperparameters, expected):
        result = list(hyperparameters.to_range())

        assert result == expected

    def test_apply(self, hyperparameters):
        result = hyperparameters.apply(hdbscan.HDBSCAN, 2)

        assert isinstance(result, hdbscan.HDBSCAN)
