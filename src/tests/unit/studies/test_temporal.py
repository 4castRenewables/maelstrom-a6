import a6.studies.temporal as temporal
import pytest


def test_perform_temporal_range_study(ds2, coordinates):
    temporal.perform_temporal_range_study(
        data=ds2,
        n_components=2,
        min_cluster_size=2,
        coordinates=coordinates,
        log_to_mantik=False,
    )


@pytest.mark.parametrize(
    ("data", "n_components", "expected"),
    [
        ("ds", 2, [slice(None, 2), slice(None, 4), slice(None, None)]),
        # PCA with 3 components requires at least 3 time steps.
        ("ds", 3, [slice(None, 4), slice(None, None)]),
        ("ds2", 2, [slice(None, 2), slice(None, 4), slice(None, None)]),
        # PCA with 3 components requires at least 3 time steps.
        ("ds2", 3, [slice(None, 4), slice(None, None)]),
    ],
)
def test_create_logarithmic_time_slices(
    request, coordinates, data, n_components, expected
):
    d = request.getfixturevalue(data)
    result = temporal._create_logarithmic_time_slices(
        d, n_components=n_components, coordinates=coordinates
    )

    assert list(result) == expected
