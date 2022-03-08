import pytest
import xarray as xr
import lifetimes.modes.methods.varimax as varimax


@pytest.fixture(scope="session")
def varimax_instance(create_unity_transformation_dataset):
    trafo_as_dataset = create_unity_transformation_dataset()
    return varimax.Varimax(trafo_as_dataset)


def test_varimax(varimax_instance):
    assert varimax_instance.as_dataset is not None
    assert varimax_instance.matrix is not None


def test_perform_varimax_rotation(
    varimax_instance, create_unity_transformation_dataset
):
    trafo_as_dataset = create_unity_transformation_dataset()
    varimax_output = varimax.perform_varimax_rotation(
        trafo_as_dataset["transformation_matrix"]
    )
    xr.testing.assert_equal(
        varimax_output.as_dataset["transformation_matrix"],
        varimax_instance.as_dataset["transformation_matrix"],
    )
