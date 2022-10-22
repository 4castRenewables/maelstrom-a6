import a6.studies.cluster as cluster
import hdbscan
import pytest


@pytest.fixture(params=[True, False])
def vary_data_variables(request) -> bool:
    return request.param


def test_perform_pca_and_cluster_hyperparameter_study(
    ds2, coordinates, hyperparameters, vary_data_variables
):
    cluster.perform_pca_and_cluster_hyperparameter_study(
        data=ds2,
        hyperparameters=hyperparameters,
        algorithm=hdbscan.HDBSCAN,
        coordinates=coordinates,
        vary_data_variables=vary_data_variables,
        log_to_mantik=False,
    )
