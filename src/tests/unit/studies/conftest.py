import a6.studies.hyperparameters as _hyperparameters
import pytest


@pytest.fixture(scope="session")
def hyperparameters() -> _hyperparameters.HyperParameters:
    return _hyperparameters.HyperParameters(
        n_components_start=1,
        n_components_end=2,
        min_cluster_size_start=2,
        min_cluster_size_end=3,
    )
