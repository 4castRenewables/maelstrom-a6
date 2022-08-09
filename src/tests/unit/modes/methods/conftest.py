import lifetimes.modes.methods.pca as _pca
import pytest
from sklearn import decomposition


@pytest.fixture(params=[decomposition.PCA, decomposition.IncrementalPCA])
def method(request):
    return request.param


@pytest.fixture()
def pcas(da, method):
    return _pca.spatio_temporal_principal_component_analysis(
        data=da,
        time_coordinate="time",
        latitude_coordinate="lat",
        x_coordinate="lat",
        y_coordinate="lon",
        variance_ratio=None,
        pca_method=method,
    )
