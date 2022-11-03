import datetime
import pathlib

import a6.datasets.coordinates as _coordinates
import a6.modes.methods.appearances as appearances
import a6.modes.methods.clustering as clustering
import a6.modes.methods.pca as _pca
import a6.testing as testing
import hdbscan as _hdbscan
import pytest
import sklearn.cluster as cluster
import xarray as xr

DATA_DIR = pathlib.Path(__file__).parent / "../data"


@pytest.fixture(scope="session")
def ml_path() -> pathlib.Path:
    return DATA_DIR / "ml_20190101_00.nc"


@pytest.fixture(scope="session")
def ml_ds(ml_path) -> xr.Dataset:
    return xr.open_dataset(ml_path)


@pytest.fixture(scope="session")
def pl_path() -> pathlib.Path:
    return DATA_DIR / "pl_20201201_00.nc"


@pytest.fixture(scope="session")
def pl_ds(pl_path) -> xr.Dataset:
    return xr.open_dataset(pl_path)


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
def coordinates() -> _coordinates.Coordinates:
    return _coordinates.Coordinates(
        time="time",
        latitude="lat",
        longitude="lon",
    )


@pytest.fixture(scope="session")
def single_variable_pca(da, coordinates) -> _pca.PCA:
    return _pca.spatio_temporal_pca(
        da,
        coordinates=coordinates,
        x_coordinate="lon",
        y_coordinate="lat",
    )


@pytest.fixture(scope="session")
def multi_variable_pca(ds2, coordinates) -> _pca.PCA:
    return _pca.spatio_temporal_pca(
        ds2,
        coordinates=coordinates,
        x_coordinate="lon",
        y_coordinate="lat",
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


@pytest.fixture()
def mode_appearances() -> appearances.Modes:
    time_delta = datetime.timedelta(days=1)

    return appearances.Modes(
        [
            appearances.Mode(
                label=0,
                appearances=[
                    appearances.Appearance(
                        label=0,
                        start=datetime.datetime(2000, 1, 1),
                        end=datetime.datetime(2000, 1, 3),
                        time_delta=time_delta,
                        index=appearances.AppearanceIndex(
                            label=0, start=0, end=2
                        ),
                    ),
                ],
                statistics=appearances.Statistics(
                    label=0,
                    abundance=1,
                    duration=appearances.Duration(
                        label=0,
                        total=datetime.timedelta(days=3),
                        max=datetime.timedelta(days=3),
                        min=datetime.timedelta(days=3),
                        mean=datetime.timedelta(days=3),
                        std=datetime.timedelta(),
                        median=datetime.timedelta(days=3),
                    ),
                ),
            ),
            appearances.Mode(
                label=1,
                appearances=[
                    appearances.Appearance(
                        label=1,
                        start=datetime.datetime(2000, 1, 4),
                        end=datetime.datetime(2000, 1, 6),
                        time_delta=time_delta,
                        index=appearances.AppearanceIndex(
                            label=1, start=3, end=5
                        ),
                    ),
                ],
                statistics=appearances.Statistics(
                    label=1,
                    abundance=1,
                    duration=appearances.Duration(
                        label=1,
                        total=datetime.timedelta(days=3),
                        max=datetime.timedelta(days=3),
                        min=datetime.timedelta(days=3),
                        mean=datetime.timedelta(days=3),
                        std=datetime.timedelta(),
                        median=datetime.timedelta(days=3),
                    ),
                ),
            ),
        ]
    )
