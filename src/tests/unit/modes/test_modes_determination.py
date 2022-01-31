import functools

from lifetimes import modes as _modes
from lifetimes import testing


def test_determine_modes():
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
    modes = [_modes.Modes(feature=ds["ellipse"])]
    pca_partial_method = functools.partial(
        _modes.methods.spatio_temporal_principal_component_analysis,
        time_coordinate="time",
        x_coordinate="lon",
        y_coordinate="lat",
        variance_ratio=None,
    )
    pca = _modes.determine_modes(modes=modes, method=pca_partial_method)

    assert pca
