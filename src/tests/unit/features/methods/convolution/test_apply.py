import a6.features.methods.convolution.apply as apply
import numpy as np
import xarray as xr


def test_apply_kernel_to_grid_1():
    kernel = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
    )

    lon = [-2.0, -1.0]
    lat = [0.0, 1.0]
    data = [
        [1.0, 1.0],
        [2.0, 2.0],
    ]
    grid = xr.DataArray(
        data=data,
        dims=["lat", "lon"],
        coords=dict(
            lon=(["lon"], lon),
            lat=(["lat"], lat),
        ),
    )
    expected_lon = lon
    expected_lat = lat
    avg = 6.0 / 4.0
    expected_data = [
        [avg, avg],
        [avg, avg],
    ]
    expected = xr.DataArray(
        data=expected_data,
        dims=["lat", "lon"],
        coords=dict(
            lat=(["lat"], expected_lat),
            lon=(["lon"], expected_lon),
        ),
    )

    result = apply.apply_kernel_to_grid(kernel, grid)

    xr.testing.assert_equal(result, expected)


def test_apply_kernel_to_grid_2():
    kernel = np.array(
        [
            [1.0],
            [1.0],
            [1.0],
        ],
    )

    lon = [-2.0, -1.0, 0.0, 1.0, 2.0]
    lat = [0.0, 1.0, 2.0, 3.0]
    data = [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0, 4.0, 4.0],
    ]
    grid = xr.DataArray(
        data=data,
        dims=["lat", "lon"],
        coords=dict(
            lat=(["lat"], lat),
            lon=(["lon"], lon),
        ),
    )
    expected_lon = lon
    expected_lat = lat
    avg = 6.0 / 3.0
    avg_2 = 4.0 / 1.0
    expected_data = [
        [avg, avg, avg, avg, avg],
        [avg, avg, avg, avg, avg],
        [avg, avg, avg, avg, avg],
        [avg_2, avg_2, avg_2, avg_2, avg_2],
    ]
    expected = xr.DataArray(
        data=expected_data,
        dims=["lat", "lon"],
        coords=dict(
            lat=(["lat"], expected_lat),
            lon=(["lon"], expected_lon),
        ),
    )

    result = apply.apply_kernel_to_grid(kernel, grid)

    xr.testing.assert_equal(result, expected)


def test_apply_kernel_to_grid_3():
    kernel = np.array(
        [
            [0.36787944, 0.60653066, 0.36787944],
            [0.60653066, 1.0, 0.60653066],
            [0.36787944, 0.60653066, 0.36787944],
        ],
    )

    lon = [-2.0, -1.0, 0.0]
    lat = [0.0, 1.0, 2.0]
    data = [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
    ]
    grid = xr.DataArray(
        data=data,
        dims=["lat", "lon"],
        coords=dict(
            lat=(["lat"], lat),
            lon=(["lon"], lon),
        ),
    )
    expected_lon = lon
    expected_lat = lat
    avg = 6.0 / 3.0
    expected_data = [
        [avg, avg, avg],
        [avg, avg, avg],
        [avg, avg, avg],
    ]
    expected = xr.DataArray(
        data=expected_data,
        dims=["lat", "lon"],
        coords=dict(
            lat=(["lat"], expected_lat),
            lon=(["lon"], expected_lon),
        ),
    )

    result = apply.apply_kernel_to_grid(kernel, grid)

    xr.testing.assert_allclose(result, expected)


def test_apply_average_kernel():
    lon = [-2.0, -1.0]
    lat = [1.0, 0.0]
    data = [
        [1.0, 1.0],
        [2.0, 2.0],
    ]
    grid = xr.DataArray(
        data=data,
        dims=["lat", "lon"],
        coords=dict(
            lat=(["lat"], lat),
            lon=(["lon"], lon),
        ),
    )
    expected_lon = [-1.5]
    expected_lat = [0.5]
    avg = 6.0 / 4.0
    expected_data = [
        [avg],
    ]
    expected = xr.DataArray(
        data=expected_data,
        dims=["lat", "lon"],
        coords=dict(
            lat=(["lat"], expected_lat),
            lon=(["lon"], expected_lon),
        ),
    )

    result = apply.apply_average_kernel(
        grid,
        latitudinal_size=1.0,
        longitudinal_size=1.0,
        latitude="lat",
        longitude="lon",
    )

    xr.testing.assert_equal(result, expected)
