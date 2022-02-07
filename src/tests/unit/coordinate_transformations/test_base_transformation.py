import pytest

import numpy as np
import xarray as xr

import lifetimes.coordinate_transformations.base_transformation as base_trafo


def test_base_transformation(create_unity_transformation_dataset):
    trafo_as_ds = create_unity_transformation_dataset()
    trafo = base_trafo.BaseTransformation(trafo_as_ds)
    np.testing.assert_equal(trafo.eigenvalues, list(range(4)))
    np.testing.assert_equal(trafo.matrix, np.eye(4).reshape(-1, 2, 2))
    xr.testing.assert_equal(trafo.as_dataset, trafo_as_ds)


def test_transform(create_unity_transformation_dataset, dataset):
    trafo_as_ds = create_unity_transformation_dataset()
    transformation = base_trafo.BaseTransformation(trafo_as_ds)
    transformed = transformation.transform(dataset, target_variable="data")
    np.testing.assert_equal(transformed["data"].values, [[1], [2], [3], [4]])
