import typing as t

import xarray as xr

import lifetimes.coordinate_transformations as transformations


class InvertibleTransformation(
    transformations.abstract_transformation.AbstractCoordinateTransformation
):
    """Wrap transformation along with inverse transformation in one class."""

    def __init__(
        self,
        transformation_as_dataset: xr.Dataset,
        inverse_transformation_as_dataset: t.Optional[xr.Dataset] = None,
        is_semi_orthogonal: bool = False,
    ):
        super().__init__()
        self.transformation = transformations.base_transformation.BaseTransformation(
            dataset=transformation_as_dataset
        )
        self.inverse_transformation = self._set_inverse_transformation(
            inverse_transformation_as_dataset, is_semi_orthogonal
        )
        self.inverse_transformation = (
            transformations.base_transformation.BaseTransformation(
                dataset=inverse_transformation_as_dataset
            )
        )

    def _set_inverse_transformation(
        self,
        inverse_transformation_as_dataset: t.Optional[xr.Dataset],
        is_semi_orthogonal: bool,
    ):
        if is_semi_orthogonal:
            if inverse_transformation_as_dataset is not None:
                raise Warning(
                    "Inverse transformation and semi othogonal transformation"
                    " are specified. Using transposed transformation as "
                    "inverse."
                )
            self.inverse_transformation = (
                transformations.base_transformation.BaseTransformation(
                    self.transformation.as_dataset.transpose()
                )
            )
        else:
            self.inverse_transformation = (
                transformations.base_transformation.BaseTransformation(
                    inverse_transformation_as_dataset
                )
            )

    def transform(
        self,
        data: xr.Dataset,
        target_variable: str,
        n_dimensions: t.Optional[int] = None,
    ) -> xr.Dataset:
        return self.transformation.transform(data, target_variable, n_dimensions)

    def inverse_transform(
        self,
        data: xr.Dataset,
        target_variable: str,
        n_dimensions: t.Optional[int] = None,
    ) -> xr.Dataset:
        return self.inverse_transformation.transform(
            data, target_variable, n_dimensions
        )

    @property
    def as_dataset(self):
        return self.transformation.as_dataset

    @property
    def eigenvalues(self):
        return self.transformation.eigenvalues

    @property
    def matrix(self):
        return self.transformation.matrix

    @property
    def inverse_matrix(self):
        return self.inverse_transformation.matrix
