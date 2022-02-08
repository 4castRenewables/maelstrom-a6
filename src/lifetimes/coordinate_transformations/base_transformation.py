import typing as t

import numpy as np
import xarray as xr

import lifetimes.coordinate_transformations as transformations


class BaseTransformation(transformations.abstract_transformation.AbstractCoordinateTransformation):
    """Wraps a base transformation in the Linear Algebra Sense."""

    def __init__(self, dataset: t.Optional[xr.Dataset] = None):
        """

        Parameters
        ----------
        dataset: xr.Dataset with original coordinates, PCs referring to those
          coordinates, and, optionally, Eigenvalues
        """
        super().__init__()
        self._dataset = dataset

    @property
    def as_dataset(self) -> xr.Dataset:
        return self._dataset

    @property
    def matrix(self) -> xr.DataArray:
        return self.as_dataset["transformation_matrix"]

    @property
    def eigenvalues(self) -> np.ndarray:
        return self.as_dataset["eigenvalues"].values


