import typing as t
import abc
import xarray as xr

class AbstractCoordinateTransformation(abc.ABC):

    def __init__(self):
        pass

    def transform(self, data: xr.Dataset, target_variable: str) -> xr.Dataset:
        """
        Transform data by applying transformation as matrix multiplication.

        Parameters
        ----------
        data: xr.Dataset with data. Subset of coordinates must match a subset of
          `self.as_dataset.coords`.
        target_variable: Variable name to apply transformation on. All others will be
          discarded.

        Returns
        -------
            xr.Dataset with transformed entries in new basis.
        """
        coefficients = (
            self.as_dataset["transformation_matrix"]
                .dot(data[target_variable])
                .to_dataset(name=target_variable)
        )
        return coefficients


    @property
    @abc.abstractmethod
    def matrix(self) -> xr.DataArray:
        pass

    @property
    @abc.abstractmethod
    def eigenvalues(self) -> t.Optional[xr.DataArray]:
        pass

    @property
    @abc.abstractmethod
    def as_dataset(self) -> xr.Dataset:
        pass