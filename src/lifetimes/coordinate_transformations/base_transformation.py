import typing as t

import numpy as np
import xarray as xr


class BaseTransformation:
    """Wraps a base transformation in the Linear Algebra Sense."""

    def __init__(self, dataset: t.Optional[xr.Dataset] = None):
        """

        Parameters
        ----------
        dataset: xr.Dataset with original coordinates, PCs referring to those
          coordinates, and, optionally, Eigenvalues
        """
        self._dataset = dataset

    @property
    def as_dataset(self) -> xr.Dataset:
        return self._dataset

    @property
    def matrix(self) -> xr.DataArray:
        return self.as_dataset["transformation_matrix"]

    @property
    def eigenvalues(self) -> np.ndarray: # TODO Maybe catch error?
        return self.as_dataset["eigenvalues"].values

    def transform(self, data: xr.Dataset, target_variable: str) -> xr.Dataset:
        coefficients = (
            self.as_dataset["transformation_matrix"]
            .dot(data[target_variable])
            .to_dataset(name=target_variable)
        )
        return coefficients

