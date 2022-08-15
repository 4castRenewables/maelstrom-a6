import hdbscan as _hdbscan
import lifetimes.modes.methods.clustering as clustering
import lifetimes.modes.methods.pca as _pca
import lifetimes.testing as testing
import pytest
import sklearn.cluster as cluster
import xarray as xr


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def da(ds) -> xr.DataArray:
    return ds["ellipse"]


@pytest.fixture(scope="session")
def ds2(da) -> xr.Dataset:
    # Create multi-variable dataset
    return xr.Dataset(
        data_vars={"ellipse_1": da, "ellipse_2": da},
        coords=da.coords,
        attrs=da.attrs,
    )


@pytest.fixture(scope="session")
def single_variable_pca(da) -> _pca.PCA:
    return _pca.spatio_temporal_pca(
        da,
        time_coordinate="time",
        latitude_coordinate="lat",
        x_coordinate="lat",
        y_coordinate="lon",
    )


@pytest.fixture(scope="session")
def multi_variable_pca(ds2) -> _pca.PCA:
    return _pca.spatio_temporal_pca(
        ds2,
        time_coordinate="time",
        latitude_coordinate="lat",
        x_coordinate="lat",
        y_coordinate="lon",
    )


@pytest.fixture(
    params=["single_variable_pca", "multi_variable_pca"], scope="session"
)
def pca(request, single_variable_pca, multi_variable_pca) -> _pca.PCA:
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def kmeans(pca) -> clustering.KMeans:
    algorithm = cluster.KMeans(n_clusters=2)
    return clustering.find_pc_space_clusters(
        algorithm=algorithm,
        pca=pca,
        n_components=3,
    )


@pytest.fixture(scope="session")
def hdbscan(pca) -> clustering.HDBSCAN:
    algorithm = _hdbscan.HDBSCAN(min_cluster_size=2)
    return clustering.find_pc_space_clusters(
        algorithm=algorithm,
        pca=pca,
        n_components=3,
    )
