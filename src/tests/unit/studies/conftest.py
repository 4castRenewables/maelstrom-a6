import a6.studies.hyperparameters as _hyperparameters
import pytest


@pytest.fixture(scope="session")
def hyperparameters() -> _hyperparameters.HyperParameters:
    return _hyperparameters.HyperParameters(
        n_components_start=1,
        n_components_end=2,
        cluster_arg="min_cluster_size",
        cluster_start=2,
        cluster_end=3,
    )
