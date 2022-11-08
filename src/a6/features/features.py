from collections.abc import Callable

import a6.datasets.ecmwf_ifs_hres as datasets
import xarray as xr

FeatureGenerator = Callable[[xr.DataArray, ...], xr.DataArray]


class Feature:
    """A feature that can be generated from any dataset."""

    def __init__(
        self,
        name: str,
        variables: list[str],
        generator: FeatureGenerator | None = None,
    ):
        """Initialize without generating the feature.

        Parameters
        ----------
        name : str
            Name of the resulting feature.
        variables : list[str]
            Name of the variables in the Dataset to use in `method`.
        generator : Callable
            Method to use for generating the feature.

        """
        if len(variables) == 0:
            raise ValueError(
                "Unable to create a feature when no variable is given"
            )
        elif generator is None and len(variables) > 1:
            raise ValueError(
                "Unable to create feature from multiple variables "
                f"{variables} without a method"
            )

        self.name = name
        self.variables = variables
        self.generator = generator

    def generate_from(self, dataset: datasets.EcmwfIfsHres) -> xr.DataArray:
        """Generate the feature from a given dataset."""
        data = dataset.to_xarray()
        result = self._generate_feature_from_dataset(data)
        result.name = self.name
        return result

    def _generate_feature_from_dataset(
        self, dataset: xr.Dataset
    ) -> xr.DataArray:
        self._check_dataset_for_missing_variables(dataset)
        if self.generator is None:
            return self._get_feature_from_dataset(dataset)
        return self._apply_feature_generator_on_dataset(dataset)

    def _get_feature_from_dataset(self, dataset: xr.Dataset) -> xr.DataArray:
        [variable] = self.variables
        return dataset[variable]

    def _apply_feature_generator_on_dataset(
        self, dataset: xr.Dataset
    ) -> xr.DataArray:
        args = (dataset[variable] for variable in self.variables)
        return self.generator(*args)

    def _check_dataset_for_missing_variables(self, dataset: xr.Dataset) -> None:
        missing = set(self.variables) - set(dataset.data_vars)
        if missing:
            raise ValueError(f"Dataset is missing required variables {missing}")
