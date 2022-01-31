from contextlib import nullcontext as doesnotraise

import pytest
import xarray as xr
from lifetimes import features
from lifetimes import testing


@pytest.fixture()
def dataset() -> features.Dataset:
    grid = testing.TestGrid(rows=3, columns=3)
    a = 2 / 3
    b = 2 / 3
    ellipse_1 = testing.EllipticalDataFactory(
        a=a,
        b=b,
    )
    data_points = [
        testing.DataPoints(
            data_factory=ellipse_1,
            start="2000-01-01",
            end="2000-01-01",
            frequency="1d",
        )
    ]
    return testing.FakeEcmwfIfsHresDataset(
        grid=grid,
        start="2000-01-01",
        end="2000-01-02",
        frequency="1d",
        data=data_points,
    )


@pytest.mark.parametrize(
    ("features_", "expected"),
    [
        (
            [
                features.Feature(
                    name="ellipse",
                    variables=["ellipse"],
                    generator=testing.methods.dummy_method,
                )
            ],
            None,
        ),
        ([features.Feature(name="ellipse", variables=["ellipse"])], None),
        (
            [features.Feature(name="ellipse", variables=["missing_variable"])],
            ValueError(),
        ),
    ],
)
def test_determine_features(dataset, features_, expected):
    if expected is None:
        expected = dataset.as_xarray()["ellipse"]

    with pytest.raises(type(expected)) if isinstance(
        expected, Exception
    ) else doesnotraise():
        [result] = features.determine_features(dataset, features_)
        xr.testing.assert_equal(result, expected)
