import logging
import pathlib

import a6
import hdbscan

logger = logging.getLogger(__name__)
a6.utils.log_to_stdout()

if __name__ == "__main__":
    data = a6.cli.data.read(
        path=pathlib.Path("/home/fabian/Documents/MAELSTROM/data/pca"),
        pattern="pl_*.nc",
        slice_files=True,
        level=500,
    )

    coordinates = a6.datasets.coordinates.Coordinates()
    variables = a6.datasets.variables.Model()

    preprocessing = (
        a6.datasets.methods.select.select_dwd_area()
        >> a6.features.methods.weighting.weight_by_latitudes(
            latitudes=coordinates.latitude,
            use_sqrt=True,
        )
        >> a6.features.methods.geopotential.calculate_geopotential_height(
            variables=variables,
        )
        >> a6.features.methods.wind.calculate_wind_speed()
        >> a6.features.methods.variables.drop_variables(
            [variables.z, variables.u, variables.v]
        )
        >> a6.features.methods.convolution.apply_kernel(
            kernel="mean",
            size=11,
            coordinates=coordinates,
        )
        >> a6.features.methods.pooling.apply_pooling(size=10, mode="mean")
    )

    hyperparameters = a6.studies.HyperParameters(
        cluster_arg="min_cluster_size",
        n_components_start=3,
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
