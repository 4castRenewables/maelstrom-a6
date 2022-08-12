import typing as t

import lifetimes.modes.methods.pca.pca_abc as pca_abc
import sklearn.decomposition as decomposition

PCAMethod = t.Union[decomposition.PCA, decomposition.IncrementalPCA]


class SingleVariablePCA(pca_abc.PCA):
    """Wrapper for `sklearn.decomposition.PCA` for single-variable data."""
