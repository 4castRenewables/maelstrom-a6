import pytest

import numpy as np
import xarray as xr


@pytest.fixture(scope="session")
def create_unity_transformation_dataset():
    def unity_transformation_dataset(
        x_coordinates=list(range(2)),
        x_coordinate_name="x",
        y_coordinates=list(range(2)),
        y_coordinate_name="y",
    ):
        dataset = xr.Dataset(
            data_vars={
                "transformation_matrix": (
                    ["eigenvector_number", x_coordinate_name, y_coordinate_name],
                    np.eye(len(x_coordinates) * len(y_coordinates)).reshape(
                        -1, len(x_coordinates), len(y_coordinates)
                    ),
                ),
                "eigenvalues": (
                    ["eigenvector_number"],
                    list(range(len(x_coordinates) * len(y_coordinates))),
                ),
            },
            coords={
                "eigenvector_number": list(
                    range(len(x_coordinates) * len(y_coordinates))
                ),
                x_coordinate_name: x_coordinates,
                y_coordinate_name: y_coordinates,
            },
        )
        return dataset

    return unity_transformation_dataset


@pytest.fixture(scope="session")
def dataset(
    x_coordinates=list(range(2)),
    x_coordinate_name="x",
    y_coordinates=list(range(2)),
    y_coordinate_name="y",
    timelike_coordinates=[1],
    timelike_coordinate_name="timelike",
    data_variable_name="data",
):
    len_x = len(x_coordinates)
    len_y = len(y_coordinates)
    len_time = len(timelike_coordinates)
    data_variables = np.array(list(range(len_x * len_y * len_time))).reshape(
        len_time, len_x, len_y
    )
    return xr.Dataset(
        data_vars={
            data_variable_name: (
                [timelike_coordinate_name, x_coordinate_name, y_coordinate_name],
                [[[1.0, 2.0], [3.0, 4.0]]],
            )
        },
        coords={
            timelike_coordinate_name: timelike_coordinates,
            "x": x_coordinates,
            "y": y_coordinates,
        },
    )
