import xarray as xr

from . import datasets
from . import features as _features


def determine_features(
    dataset: datasets.Dataset, features: list[_features.Feature]
) -> list[xr.DataArray]:
    """Determine a set of features from a dataset."""
    return [feature.generate_from(dataset) for feature in features]
