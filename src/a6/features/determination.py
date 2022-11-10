import xarray as xr

import a6.datasets.ecmwf_ifs_hres as datasets
import a6.features.features as _features


def determine_features(
    dataset: datasets.EcmwfIfsHres, features: list[_features.Feature]
) -> list[xr.DataArray]:
    """Determine a set of features from a dataset."""
    return [feature.generate_from(dataset) for feature in features]
