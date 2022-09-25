import a6.studies.pca_and_hdbscan as pca_and_hdbscan
import pytest


@pytest.fixture(params=[True, False])
def vary_data_variables(request) -> bool:
    return request.param


def test_perform_pca_and_hdbscan_hyperparameter_study(
    ds2, coordinates, hyperparameters, vary_data_variables
):
    pca_and_hdbscan.perform_pca_and_hdbscan_hyperparameter_study(
        data=ds2,
        hyperparameters=hyperparameters,
        coordinates=coordinates,
        vary_data_variables=vary_data_variables,
        log_to_mantik=False,
    )
