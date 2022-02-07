import typing as t

import xarray as xr

import lifetimes.coordinate_transformations.base_transformation as base_transformation


class InvertibleTransformation:
    """Wrap transformation along with inverse transformation in one class."""

    def __init__(
        self,
        transformation_as_dataset: t.Optional[xr.Dataset] = None,
        inverse_transformation_as_dataset: t.Optional[xr.Dataset] = None,
    ):
        self.transformation = base_transformation.BaseTransformation(
            dataset=transformation_as_dataset
        )
        self.inverse_transformation = base_transformation.BaseTransformation(
            dataset=inverse_transformation_as_dataset
        )

    def transform(self, data: xr.Dataset, target_variable):
        return self.transformation.transform(data, target_variable)

    def inverse_transform(self, data: xr.Dataset, target_variable):
        return self.inverse_transformation.transform(data, target_variable)

    @property
    def matrix(self):
        return self.transformation.matrix

    @property
    def inverse_matrix(self):
        return self.inverse_transformation.matrix

    @classmethod
    def from_SO_N(cls, so_n_transformation_as_dataset: xr.Dataset):
        """
        Create class from SO(N) transformation satisfying, in matrix notation, X^T*X=1.

        Parameters
        ----------
        so_n_transformation_as_dataset: Transformation Matrix and optionally eigenvalues
          as xarray dataset.

        Returns
        -------
            Class instance.
        """
        inverse_transformation_as_dataset = so_n_transformation_as_dataset.transpose()
        return cls(so_n_transformation_as_dataset, inverse_transformation_as_dataset)
