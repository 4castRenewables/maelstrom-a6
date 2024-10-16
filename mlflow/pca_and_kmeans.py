import logging
import pathlib

import hdbscan

import a6


if __name__ == "__main__":
    a6.utils.log_to_stdout()
    logger = logging.getLogger(__name__)

    kernel_size = 11
    kernel_mode = "mean"

    coordinates = a6.datasets.coordinates.Coordinates()
    variables = a6.datasets.variables.Model()

    preprocessing = a6.datasets.methods.select.select_dwd_area(
        coordinates=coordinates
    ) >> a6.datasets.methods.select.select_levels_and_calculate_daily_mean(
        levels=500
    )

    data = a6.datasets.EcmwfIfsHres(
        path=pathlib.Path(
            # "/home/fabian/Documents/data/pca"
            "/p/scratch1/deepacf/maelstrom_data/a6/pl"
        ),
        pattern="pl_*.nc",
        slice_time_dimension=True,
        preprocessing=preprocessing,
        postprocessing=a6.features.methods.averaging.calculate_daily_mean(
            coordinates=coordinates
        ),
    ).to_xarray()

    preprocessing = (
        a6.features.methods.weighting.weight_by_latitudes(
            latitudes=coordinates.latitude,
            use_sqrt=True,
        )
        >> a6.features.methods.geopotential.calculate_geopotential_height(
            variables=variables,
        )
        >> a6.features.methods.wind.calculate_wind_speed(variables=variables)
        >> a6.features.methods.variables.drop_variables(
            names=[variables.z, variables.u, variables.v]
        )
        >> a6.features.methods.convolution.apply_kernel(
            kernel=kernel_mode,
            size=kernel_size,
            coordinates=coordinates,
        )
        >> a6.features.methods.pooling.apply_pooling(
            size=kernel_size, mode=kernel_mode, coordinates=coordinates
        )
    )

    hyperparameters = a6.studies.HyperParameters(
        cluster_arg="min_cluster_size",
        n_components_start=2,
        n_components_end=12,
        cluster_start=2,
        cluster_end=30,
    )
    a6.studies.perform_pca_and_cluster_hyperparameter_study(
        data=preprocessing.apply_to(data),
        algorithm=hdbscan.HDBSCAN,
        hyperparameters=hyperparameters,
        coordinates=coordinates,
        use_varimax=False,
        vary_data_variables=False,
        log_to_mantik=True,
    )
