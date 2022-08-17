import lifetimes.hyperparameters as _hyperparameters
import pytest


class TestHyperParamers:
    @pytest.mark.parametrize(
        ("hyperparameters", "expected"),
        [
            (
                _hyperparameters.HyperParameters(
                    n_components_start=1,
                    min_cluster_size_start=2,
                    min_cluster_size_end=3,
                ),
                [(1, 2), (1, 3)],
            ),
            (
                _hyperparameters.HyperParameters(
                    n_components_start=1,
                    n_components_end=2,
                    min_cluster_size_start=2,
                ),
                [(1, 2), (2, 2)],
            ),
            (
                _hyperparameters.HyperParameters(
                    n_components_start=1,
                    n_components_end=2,
                    min_cluster_size_start=2,
                    min_cluster_size_end=3,
                ),
                [(1, 2), (1, 3), (2, 2), (2, 3)],
            ),
        ],
    )
    def test_to_range(self, hyperparameters, expected):
        result = list(hyperparameters.to_range())

        assert result == expected
