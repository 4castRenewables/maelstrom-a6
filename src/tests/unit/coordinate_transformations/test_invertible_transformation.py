import numpy as np
import pytest

import lifetimes.coordinate_transformations.invertible_transformation as invertible_transformation_


@pytest.fixture(scope="session")
def invertible_transformation(create_unity_transformation_dataset):
    transformation_dataset = create_unity_transformation_dataset()
    return invertible_transformation_.InvertibleTransformation(
        transformation_dataset, transformation_dataset
    )


def test_invertible_transformation(invertible_transformation):
    np.testing.assert_equal(
        invertible_transformation.matrix.values.reshape(-1, 4, 4), [np.eye(4)]
    )
    np.testing.assert_equal(
        invertible_transformation.inverse_matrix.values.reshape(-1, 4, 4), [np.eye(4)]
    )


def test_transformation(invertible_transformation, dataset):
    transformed = invertible_transformation.transform(dataset, target_variable="data")
    np.testing.assert_equal(transformed["data"].values, [[1], [2], [3], [4]])


def test_inverse_transformation(invertible_transformation, dataset):
    transformed = invertible_transformation.transform(dataset, target_variable="data")
    inverse_transformed = invertible_transformation.inverse_transform(
        transformed, target_variable="data"
    )
    assert inverse_transformed.broadcast_equals(dataset)


def test_from_SO_N(create_unity_transformation_dataset, dataset):
    transformation_as_dataset = create_unity_transformation_dataset()
    transformation = invertible_transformation_.InvertibleTransformation.from_SO_N(
        transformation_as_dataset
    )
    # Test roundtrip
    transformed = transformation.transform(dataset, target_variable="data")
    inverse_transformed = transformation.inverse_transform(
        transformed, target_variable="data"
    )
    assert inverse_transformed.broadcast_equals(dataset)
