import lifetimes.features.datasets as datasets
import lifetimes.features.features as _features
import xarray as xr


def determine_features(
    dataset: datasets.Dataset, features: list[_features.Feature]
) -> list[xr.DataArray]:
    """Determine a set of features from a dataset."""
    return [feature.generate_from(dataset) for feature in features]
