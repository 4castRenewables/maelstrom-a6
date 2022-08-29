import lifetimes.studies.pca_and_kmeans as pca_and_kmeans
import pytest


@pytest.mark.parametrize("data", ["ds", "ds2"])
def test_perform_pca_and_kmeans(request, coordinates, data):
    data = request.getfixturevalue(data)

    pca_and_kmeans.perform_pca_and_kmeans(
        data=data,
        n_components=3,
        n_clusters=3,
        coordinates=coordinates,
        use_varimax=False,
        log_to_mantik=False,
    )
