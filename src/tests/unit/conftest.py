import hdbscan as _hdbscan
import lifetimes.modes.methods.clustering as clustering
import lifetimes.modes.methods.pca as _pca
import lifetimes.testing as testing
import pytest
import xarray as xr
from sklearn import cluster
from sklearn import decomposition


@pytest.fixture()
def ds() -> xr.Dataset:
    grid = testing.TestGrid(rows=10, columns=10)
    ellipse_1 = testing.EllipticalDataFactory(
        a=0.15,
        b=0.25,
        center=(-0.5, -0.5),
    )
    ellipse_2 = testing.EllipticalDataFactory(
        a=0.15,
        b=0.25,
        center=(0.5, 0.5),
    )
    data_points = [
        testing.DataPoints(
            data_factory=ellipse_1,
            start="2000-01-01",
            end="2000-01-02",
            frequency="1d",
        ),
        testing.DataPoints(
            data_factory=ellipse_2,
            start="2000-01-01",
            end="2000-01-03",
            frequency="1d",
        ),
    ]
    dataset = testing.FakeEcmwfIfsHresDataset(
        grid=grid,
        start="2000-01-01",
        end="2000-01-05",
        frequency="1d",
        data=data_points,
    )
    ds = dataset.as_xarray()
    return ds


@pytest.fixture()
def da(ds) -> xr.DataArray:
    return ds["ellipse"]


@pytest.fixture()
def pca(da):
    return _pca.spatio_temporal_principal_component_analysis(
        data=da,
        time_coordinate="time",
        latitude_coordinate="lat",
        x_coordinate="lat",
        y_coordinate="lon",
        variance_ratio=None,
        pca_method=decomposition.PCA,
    )


@pytest.fixture()
def kmeans(pca) -> clustering.KMeans:
    algorithm = cluster.KMeans(n_clusters=2)
    return clustering.find_principal_component_clusters(
        algorithm=algorithm,
        pca=pca,
        n_components=3,
    )


@pytest.fixture()
def hdbscan(pca) -> clustering.HDBSCAN:
    algorithm = _hdbscan.HDBSCAN(min_cluster_size=2)
    return clustering.find_principal_component_clusters(
        algorithm=algorithm,
        pca=pca,
        n_components=3,
    )
