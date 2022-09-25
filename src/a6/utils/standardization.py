import a6.utils._types as _types
import numpy as np


def standardize(data: _types.Data) -> _types.Data:
    """Standardize to zero mean and unit variance (standard deviation).

    Standardizing to zero mean and unit variance is important when using more
    than 1 variable.

    """
    mean_subtracted = data - np.nanmean(data)
    standardized = mean_subtracted / np.nanstd(mean_subtracted)
    return standardized
